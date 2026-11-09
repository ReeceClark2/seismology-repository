import csv
import multiprocessing as mp
import os
import random
from time import perf_counter
from datetime import datetime
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

# Configure Matplotlib before importing pyplot.
os.environ["MPLBACKEND"] = "Agg"

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import matplotlib
from matplotlib.patches import Ellipse

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

jax.config.update("jax_enable_x64", True)


# Dataclasses ----------------------------------------------------------------------------

@dataclass
class GridSearchTask:
    f_min: float
    f_max: float
    Q_min: float
    Q_max: float
    f_points: int
    Q_points: int

# Data Acquisition -----------------------------------------------------------------------

def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, f_min, f_max):
    """Download, response-correct, decimate, and broadly bandpass the data."""
    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)
    print(network, station, channel, location, stream_index, start_time, end_time)
    client = Client("EarthScope")

    inventory = client.get_stations(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=start_time,
        endtime=end_time,
        level="response",
    )

    stream = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=start_time,
        endtime=end_time,
    )

    trace = stream[stream_index]
    trace.detrend("spline", order=3, dspline=1728)
    trace.remove_response(inventory=inventory, output="ACC")

    trace.decimate(5, no_filter=True)
    trace.decimate(5, no_filter=True)
    trace.decimate(3, no_filter=True)

    trace.filter("bandpass", freqmin=f_min, freqmax=f_max, corners=2)

    delta = float(trace.stats.delta)
    sample_count = len(trace)

    t = np.arange(sample_count, dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)

    return t, d

# Signal Update --------------------------------------------------------------------------

def get_log_prob(t, d, B, signals):
    '''
    Bretthorst Eq. 3.17
    '''
    
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi

    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)
    # g = g + 1e-8 * jnp.eye(g.shape[0], dtype=g.dtype)

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    ratio = sum_sq_proj / sum_sq_data

    return 0.5 * (m - N) * jnp.log1p(-ratio)

def grid_search(t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, batch_size=512):
    f_space = jnp.linspace(f_min, f_max, f_points)
    B = f_max - f_min

    Q_margin = 1e-6 * (Q_max - Q_min)
    Q_space = jnp.geomspace(Q_min + Q_margin, Q_max - Q_margin, Q_points)

    f_grid, Q_grid = jnp.meshgrid(f_space, Q_space, indexing="ij")
    signal_grid = jnp.stack([f_grid.ravel(), Q_grid.ravel()], axis=-1)

    evaluate_batch = jax.jit(
        jax.vmap(lambda signal: get_log_prob(t, d, B, signal))
    )

    results = []
    
    for start in range(0, signal_grid.shape[0], batch_size):
        batch = signal_grid[start:start + batch_size]
        results.append(evaluate_batch(batch))

    log_probs = jnp.concatenate(results)
    log_prob_space = log_probs.reshape(f_space.size, Q_space.size)

    probability_surface = (f_space, Q_space, log_prob_space)
    
    return probability_surface

def objective(t, d, B, f_min, f_max, Q_min, Q_max):
    fs = numpyro.sample("fs", numpyro.distributions.Uniform(f_min, f_max).to_event(1))
    log_Qs = numpyro.sample("log_Qs", numpyro.distributions.Uniform(jnp.log(Q_min), jnp.log(Q_max)).to_event(1))

    Qs = numpyro.deterministic("Qs", jnp.exp(log_Qs))
    signals = jnp.stack((fs, Qs), axis=-1)

    numpyro.factor("surface", get_log_prob(t, d, B, signals))
 
def nuts(t, d, B, signals, bounds, warmup, sample, chains, rng_key_value=None):
    f_init, Q_init = unpack_signals(signals, sort=False)
    k_init = jnp.pi * f_init / Q_init

    bounds = jnp.asarray(bounds).reshape(-1, 4)

    fs_min = bounds[:, 0].astype(f_init.dtype)
    fs_max = bounds[:, 1].astype(f_init.dtype)
    Qs_min = bounds[:, 2].astype(Q_init.dtype)
    Qs_max = bounds[:, 3].astype(Q_init.dtype)
    ks_min = jnp.pi * fs_min / Qs_min
    ks_max = jnp.pi * fs_max / Qs_max

    log_Q_init = jnp.log(Q_init)

    init_strategy = init_to_value(values={"fs": f_init, "log_Qs": log_Q_init})

    if rng_key_value is None:
        rng_key_value = random.randint(1, 10_000)

    rng_key = jax.random.PRNGKey(int(rng_key_value))
    warmup_key, sample_key = jax.random.split(rng_key)

    kernel = NUTS(objective, init_strategy=init_strategy, max_tree_depth=5)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=sample, num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.warmup(warmup_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, Q_min=Qs_min, Q_max=Qs_max, extra_fields=("potential_energy",))

    warmup_samples = mcmc.get_samples(group_by_chain=True)
    warmup_extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    mcmc.run(mcmc.post_warmup_state.rng_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, Q_min=Qs_min, Q_max=Qs_max, extra_fields=("potential_energy",))

    sample_samples = mcmc.get_samples(group_by_chain=True)
    sample_extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    samples = {name: jnp.concatenate([warmup_samples[name], sample_samples[name]], axis=1) for name in sample_samples}
    extra_fields = {name: jnp.concatenate([warmup_extra_fields[name], sample_extra_fields[name]], axis=1) for name in sample_extra_fields}

    potential_energy = extra_fields["potential_energy"]

    best_flat_index = jnp.argmin(potential_energy.ravel())
    chain_index, draw_index = jnp.unravel_index(best_flat_index, potential_energy.shape)

    best_fs = samples["fs"][chain_index, draw_index]
    best_Qs = jnp.exp(samples["log_Qs"][chain_index, draw_index])

    signals = jnp.stack((best_fs, best_Qs), axis=-1)
    return signals

# Orchestrators ---------------------------------------------------------------------------

def _run_grid_search(args):
    t, d, task = args

    return grid_search(
        t=t,
        d=d,
        f_min=task.f_min,
        f_max=task.f_max,
        Q_min=task.Q_min,
        Q_max=task.Q_max,
        f_points=task.f_points,
        Q_points=task.Q_points,
    )

def grid_search_orchestrator(path, signals, t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, cores):
    f_margin = 1e-6 * (f_max - f_min)
    full_f_space = np.linspace(f_min + f_margin, f_max - f_margin, f_points)

    boundaries = np.linspace(
        0, f_points, cores + 1, dtype=int
    )

    tasks = []

    for i in range(cores):
        start = boundaries[i]
        stop = boundaries[i + 1]

        tasks.append(
            GridSearchTask(
                f_min=float(full_f_space[start]),
                f_max=float(full_f_space[stop - 1]),
                Q_min=Q_min,
                Q_max=Q_max,
                f_points=stop - start,
                Q_points=Q_points,  # Do not divide this
            )
        )

    args = [(t, d, task) for task in tasks]

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=cores, mp_context=ctx) as executor:
        surfaces = list(executor.map(_run_grid_search, args))

    log_prob_space = jnp.concatenate(
        [surface[2] for surface in surfaces],
        axis=0,
    )
    f_space = jnp.concatenate(
        [surface[0] for surface in surfaces],
        axis=0,
    )
    Q_space = surfaces[0][1]

    probability_surface = (
        f_space,
        Q_space,
        log_prob_space,
    )

    plot_probability_surface(f"Probability Surface {len(signals)}", path / "diagnostic" / "probability_surface" / f"probability_surface_{len(signals)}.png", probability_surface)

    f_space, Q_space, log_prob_space = probability_surface
    best_index = jnp.unravel_index(jnp.argmax(log_prob_space), log_prob_space.shape)

    f_index, Q_index = best_index
    f_space[f_index]
    signal = (f_space[f_index], Q_space[Q_index])

    return signal

def _nuts_worker(args):
    checkpoint, t, d, B, warmup, sample, cores_per_worker, seed = args
    signals, bounds = checkpoint

    optimized_signals = nuts(
        t=t,
        d=d,
        B=B,
        signals=signals,
        bounds=bounds,
        warmup=warmup,
        sample=sample,
        chains=cores_per_worker,
        rng_key_value=seed,
    )

    return optimized_signals, bounds

def nuts_orchestrator(checkpoints, t, d, B, warmup, sample, devices):
    if not checkpoints:
        return []

    n_checkpoints = len(checkpoints)
    n_workers = min(devices, n_checkpoints)
    devices_per_worker = max(1, devices // n_checkpoints)

    tasks = [(checkpoint, t, d, B, warmup, sample, devices_per_worker, index) for index, checkpoint in enumerate(checkpoints)]
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        optimized_checkpoints = list(executor.map(_nuts_worker, tasks))

    return optimized_checkpoints

# Evaluation (critical) -------------------------------------------------------------------

def get_mean_sq_proj(t, d, B, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)

    return mean_sq_proj

def get_glob_ll(t, d, signals, f_min, f_max, Q_min, Q_max, ratio_delta=1e2, ratio_gamma=1e2, ratio_sigma=1e2):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    B = f_max - f_min
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # TODO: think about the scaling closely.
    omega_center = jnp.pi * (f_max + f_min)
    omega_halfwidth = jnp.pi * (f_max - f_min)

    log_Q_min = jnp.log(Q_min)
    log_Q_max = jnp.log(Q_max)
    log_Q_center = 0.5 * (log_Q_max + log_Q_min)
    log_Q_halfwidth = 0.5 * (log_Q_max - log_Q_min)

    x_omega = (omegas - omega_center) / omega_halfwidth
    x_log_Q = (jnp.log(Qs) - log_Q_center) / log_Q_halfwidth
    
    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)
    mean_sq_param = (1 / r) * jnp.sum(x_omega ** 2 + x_log_Q ** 2)

    log_R_delta = jnp.log(ratio_delta)
    log_R_gamma = jnp.log(ratio_gamma)
    log_R_sigma = jnp.log(ratio_sigma)

    theta, unravel = ravel_pytree(signals)
    
    def objective(theta):
        return get_mean_sq_proj(t, d, B, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues = jnp.linalg.eigvalsh(b)

    log_jacobian_factor = -0.5 * jnp.sum(jnp.log(abs(eigenvalues)))

    delta_term = (
        jsp.gammaln(m / 2)
        - jnp.log(2) - jnp.log(log_R_delta)
        + (-m / 2) * jnp.log(m * mean_sq_proj / 2.0)
    )

    gamma_term = (
        jsp.gammaln(r / 2)
        - jnp.log(2) - jnp.log(log_R_gamma)
        + (-r / 2) * jnp.log((r * mean_sq_param) / 2)
    )

    sigma_term = (
        jsp.gammaln((N - m - r) / 2)
        - jnp.log(2) - jnp.log(log_R_sigma)
        + ((m + r - N) / 2) * jnp.log(
            ((N * mean_sq_data) - (m * mean_sq_proj)) / 2
        )
    )

    glob_ll = delta_term + gamma_term + sigma_term + log_jacobian_factor
    return glob_ll

def get_condition_number(t, d, B, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    norms = jnp.sqrt(jnp.diag(g))
    g = g / jnp.outer(norms, norms)

    eigenvalues = jnp.linalg.eigvalsh(g)

    max_eigenvalue = jnp.max(eigenvalues)
    min_eigenvalue = jnp.maximum(jnp.min(eigenvalues), jnp.finfo(eigenvalues.dtype).eps * max_eigenvalue)

    # Gram eigenvalues are squared singular values.
    k2 = jnp.sqrt(max_eigenvalue / min_eigenvalue)
    return k2

def get_model(t, d, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    coefficients = jnp.linalg.lstsq(G.T, d, rcond=None)[0]

    model =  G.T @ coefficients
    return model

# Evaluation (minor) ----------------------------------------------------------------------

def get_noise_variance(t, d, B, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)
    return noise_variance

def get_snr(t, d, B, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

    snr = ((m / N) * (1 + ((1 / m) * sum_sq_proj / noise_variance))) ** (1 / 2)
    return snr

def get_phasor(t, d, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs
    
    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    beta = eigenvectors @ (h / jnp.sqrt(eigenvalues))
    n_components = fs.shape[0]

    cosine_coefficients = beta[:n_components]
    sine_coefficients = beta[n_components:]

    amplitudes = jnp.sqrt(cosine_coefficients**2 + sine_coefficients**2)
    phases = jnp.atan2(cosine_coefficients, sine_coefficients)
    return amplitudes, phases

def get_uncertainties(t, d, B, signals):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi
    
    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

    theta, unravel = ravel_pytree(signals)
    
    def objective(theta):
        return get_mean_sq_proj(t, d, B, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = jnp.linalg.eigh(b)

    variances = jnp.sqrt(noise_variance * jnp.sum(eigenvectors ** 2 / eigenvalues[None, :], axis=1))
    fs_unc = variances[::2]
    Qs_unc = variances[1::2]

    return fs_unc, Qs_unc

# Evaluation (all) ------------------------------------------------------------------------

def get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, ratio_delta=1e2, ratio_gamma=1e2, ratio_sigma=1e2):
    fs, Qs = unpack_signals(signals)
    ks = jnp.pi * fs / Qs

    omegas = fs * 2.0 * jnp.pi

    n_signals = fs.shape[0]
    r = n_signals * 2
    m = n_signals * 2
    N = int(2 * t[-1] * B)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    mean_sq_data = (1 / N) * sum_sq_data
    mean_sq_proj = (1 / m) * sum_sq_proj

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

    snr = ((m / N) * (1 + ((1 / m) * sum_sq_proj / noise_variance))) ** (1 / 2)

    g_norms = jnp.sqrt(jnp.diag(g))
    g_normalized = g / jnp.outer(g_norms, g_norms)
    g_eigenvalues = jnp.linalg.eigvalsh(g_normalized)

    max_eigenvalue = jnp.max(g_eigenvalues)
    min_eigenvalue = jnp.maximum(jnp.min(g_eigenvalues), jnp.finfo(g_eigenvalues.dtype).eps * max_eigenvalue)
    k2 = jnp.sqrt(max_eigenvalue / min_eigenvalue)

    ratio = sum_sq_proj / sum_sq_data
    log_prob = 0.5 * (m - N) * jnp.log1p(-ratio)

    omega_center = jnp.pi * (f_max + f_min)
    omega_halfwidth = jnp.pi * (f_max - f_min)

    log_Q_min = jnp.log(Q_min)
    log_Q_max = jnp.log(Q_max)
    log_Q_center = 0.5 * (log_Q_max + log_Q_min)
    log_Q_halfwidth = 0.5 * (log_Q_max - log_Q_min)

    x_omega = (omegas - omega_center) / omega_halfwidth
    x_log_Q = (jnp.log(Qs) - log_Q_center) / log_Q_halfwidth
    
    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)
    mean_sq_param = (1 / r) * jnp.sum(x_omega ** 2 + x_log_Q ** 2)

    log_R_delta = jnp.log(ratio_delta)
    log_R_gamma = jnp.log(ratio_gamma)
    log_R_sigma = jnp.log(ratio_sigma)

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, B, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    b_eigenvalues, b_eigenvectors = jnp.linalg.eigh(b)
    covariance = noise_variance * ((b_eigenvectors / b_eigenvalues[None, :]) @ b_eigenvectors.T)

    unc = jnp.sqrt(noise_variance * jnp.sum(b_eigenvectors ** 2 / b_eigenvalues[None, :], axis=1))
    fs_unc = unc[::2]
    Qs_unc = unc[1::2]

    amplitudes = jnp.sqrt(
        (eigenvectors @ (h / jnp.sqrt(eigenvalues)))[:len(signals)] ** 2
        + (eigenvectors @ (h / jnp.sqrt(eigenvalues)))[len(signals):] ** 2
    )

    beta = eigenvectors @ (h / jnp.sqrt(eigenvalues))
    cosine_coefficients = beta[:len(signals)]
    sine_coefficients = beta[len(signals):]
    amplitudes = jnp.sqrt(cosine_coefficients ** 2 + sine_coefficients ** 2)
    phases = jnp.atan2(cosine_coefficients, sine_coefficients)

    delta_term = (
        jsp.gammaln(m / 2)
        - jnp.log(2)
        - jnp.log(log_R_delta)
        + (-m / 2) * jnp.log(m * mean_sq_proj / 2.0)
    )

    gamma_term = (
        jsp.gammaln(r / 2)
        - jnp.log(2)
        - jnp.log(log_R_gamma)
        + (-r / 2) * jnp.log((r * mean_sq_param) / 2)
    )

    sigma_term = (
        jsp.gammaln((N - m - r) / 2)
        - jnp.log(2)
        - jnp.log(log_R_sigma)
        + ((m + r - N) / 2) * jnp.log(
            ((N * mean_sq_data) - (m * mean_sq_proj)) / 2
        )
    )

    log_jacobian_factor = -0.5 * jnp.sum(jnp.log(jnp.abs(b_eigenvalues)))
    glob_ll = delta_term + gamma_term + sigma_term + log_jacobian_factor

    return log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance

# Plotting --------------------------------------------------------------------------------

def save_iteration_diagnostics(path, signals, noise_variance, snr, log_prob, glob_ll, k2):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostic_path / "iteration_diagnostics.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Signals", "Noise Variance", "SNR", "Probability", "Global Likelihood", "2-norm"])
        writer.writerow([
            len(signals),
            float(noise_variance),
            float(snr),
            float(log_prob),
            float(glob_ll),
            float(k2),
        ])

def save_signal_diagnostics(path, iteration, fs, Qs, fs_unc, Qs_unc, amplitudes, phases):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostic_path / f"signals_iteration_{iteration + 1}.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency", "Frequency Uncertainty", "Quality Factor", "Quality Factor Uncertainty", "Amplitude", "Phase"])
        writer.writerows(zip(fs, fs_unc, Qs, Qs_unc, amplitudes, phases))

def plot_probability_surface(title, path, probability_surface):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f_space, Q_space, log_prob_space = probability_surface
    log_prob_space = log_prob_space.T

    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap = ax.pcolormesh(f_space, Q_space, log_prob_space, shading="auto", cmap="Grays")

    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Quality Factor (Q)")
    ax.set_title(title)

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Log Probability")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

def plot_signal_space(title, path, fs, Qs, fs_unc, Qs_unc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = np.asarray(fs)
    Qs = np.asarray(Qs)
    fs_unc = np.asarray(fs_unc)
    Qs_unc = np.asarray(Qs_unc)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(fs, Qs, color="red", s=15, zorder=3, label="Signals")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for f, q, f_unc, q_unc in zip(fs, Qs, fs_unc, Qs_unc):
        for sigma, alpha in ((3.0, 0.06), (2.0, 0.12), (1.0, 0.22)):
            ellipse = Ellipse(xy=(f, q), width=2.0 * sigma * f_unc, height=2.0 * sigma * q_unc, facecolor="red", edgecolor="red", alpha=alpha, linewidth=0.8)
            ax.add_patch(ellipse)

    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Quality Factor (Q)")
    ax.set_title(title)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

def plot_time_series(title, path, t, d, signals=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(t, d, color="black", label="Data")
    ax.set_xlim(t[0], t[-1])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)

    if signals is not None:
        model = get_model(t, d, signals)

        ax.plot(t, model, color="red", label="Model", alpha=0.6)
        ax.plot(t, d-model, color="blue", label="Residual", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

# TODO: Inspect covariance plotting
def plot_covariance(title, path, signals, covariance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    signals = np.asarray(signals)
    if signals.ndim == 1:
        signals = signals.reshape(1, 2)

    covariance = np.asarray(covariance, dtype=float)
    n_signals = signals.shape[0]
    n_parameters = 2 * n_signals

    order = np.concatenate((np.arange(0, n_parameters, 2), np.arange(1, n_parameters, 2)))
    covariance = covariance[np.ix_(order, order)]

    labels = [f"f{index + 1}" for index in range(n_signals)] + [f"Q{index + 1}" for index in range(n_signals)]
    std = np.sqrt(np.diag(covariance))
    denominator = np.outer(std, std)

    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = np.divide(covariance, denominator, out=np.full(covariance.shape, np.nan, dtype=float), where=np.isfinite(denominator) & (denominator > 0))

    finite_covariance = covariance[np.isfinite(covariance)]
    cov_limit = np.max(np.abs(finite_covariance)) if finite_covariance.size else 0.0
    if cov_limit == 0:
        cov_limit = 1.0

    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("lightgray")

    fig, (ax_cov, ax_corr) = plt.subplots(1, 2, figsize=(12, 8))

    cov_image = ax_cov.imshow(np.ma.masked_invalid(covariance), cmap=cmap, vmin=-cov_limit, vmax=cov_limit, interpolation="nearest")
    ax_cov.set_title("Covariance")
    fig.colorbar(cov_image, ax=ax_cov, label="Covariance")

    corr_image = ax_corr.imshow(np.ma.masked_invalid(correlation), cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax_corr.set_title("Correlation")
    fig.colorbar(corr_image, ax=ax_corr, label="Correlation")

    for ax in (ax_cov, ax_corr):
        ax.set_xticks(np.arange(n_parameters), labels=labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(n_parameters), labels=labels)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_fourier_space(title, path, t, d, f_min, f_max, signals=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f, p = get_fft(t, d, f_min, f_max)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(f, abs(p), color="black", label="Data", linewidth=0.9)
    ax.set_xlim(f[0], f[-1])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    if signals is not None:
        fs, Qs = unpack_signals(signals)
        ks = jnp.pi * fs / Qs
        amplitudes, phases = get_phasor(t, d, signals)

        model_p = get_model_fourier(f, fs, ks, amplitudes, phases)
        ax.plot(f, abs(model_p), color="red", alpha=0.8, label="Model", linewidth=0.9)
        ax.plot(f, abs(p) - abs(model_p), color="blue", alpha=0.6, label="Residual", linewidth=0.9)

        ymin, ymax = ax.get_ylim()
        ymin_val = -jnp.abs(jnp.min(jnp.abs(p) - jnp.abs(model_p))) * 100
        ymax_val = jnp.abs(jnp.max(jnp.abs(p) - jnp.abs(model_p))) * 100

        ax.vlines(fs, ymin_val, ymax_val, color="red", linestyle="--", alpha=0.15, linewidth=0.4)
        ax.set_ylim(ymin, ymax)

        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

def initial_plot(path, t, d, f_min, f_max):
    plot_time_series("Time Series", path / "time_series.png", t, d)
    plot_fourier_space("Fourier Space", path / "fourier_space.png", t, d, f_min, f_max)
    
    return

# Utilities -------------------------------------------------------------------------------

def unpack_signals(signals, sort=True):
    signals = jnp.asarray(signals)

    if signals.ndim == 1:
        if signals.shape[0] != 2:
            raise ValueError(
                f"A single signal must have shape (2,), got {signals.shape}"
            )
        signals = signals[None, :]

    if sort is True:
        signals = signals[jnp.argsort(signals[:, 0])]

    fs = signals[:, 0]
    Qs = signals[:, 1]

    return fs, Qs

def get_fft(t, d, f_min, f_max):
    dt = np.mean(np.diff(t))
    f = np.fft.rfftfreq(len(d), dt)
    p = dt * np.abs(np.fft.rfft(d))
    mask = (f >= f_min) & (f <= f_max)

    return f[mask], p[mask]

def get_decay_sine_fourier(f, frequency, decay_rate, amplitude, phase):
    positive_term = jnp.exp(1j * phase) / (decay_rate + 1j * 2 * jnp.pi * (f - frequency))
    negative_term = jnp.exp(-1j * phase) / (decay_rate + 1j * 2 * jnp.pi * (f + frequency))

    return amplitude * (positive_term - negative_term) / (2j)

def get_model_fourier(f, fs, ks, amplitudes, phases):
    model_p = jnp.zeros_like(f, dtype=jnp.complex128)

    for frequency, decay_rate, amplitude, phase in zip(fs, ks, amplitudes, phases):
        model_p = model_p + get_decay_sine_fourier(f, frequency, decay_rate, amplitude, phase)

    return model_p

def get_bounds(signals, f_min, f_max, Q_min, Q_max, f_fraction=1 / 8, log_Q_fraction=1 / 8):
    fs, Qs = unpack_signals(signals)

    frequency_halfwidth = (f_max - f_min) / (2 * f_fraction)

    log_Q_space_min = jnp.log(Q_min)
    log_Q_space_max = jnp.log(Q_max)
    log_Q_halfwidth = (log_Q_space_max - log_Q_space_min) / (2 * log_Q_fraction)

    frequency_min = jnp.maximum(fs - frequency_halfwidth, f_min)
    frequency_max = jnp.minimum(fs + frequency_halfwidth, f_max)

    log_Qs = jnp.log(Qs)
    log_Q_min = jnp.maximum(log_Qs - log_Q_halfwidth, log_Q_space_min)
    log_Q_max = jnp.minimum(log_Qs + log_Q_halfwidth, log_Q_space_max)

    quality_factor_min = jnp.exp(log_Q_min)
    quality_factor_max = jnp.exp(log_Q_max)

    return jnp.stack((frequency_min, frequency_max, quality_factor_min, quality_factor_max), axis=-1)

def get_shift(x, y):
    a, b, c = np.polyfit(x, y, 2)
    x_shift = -b / (2 * a)

    return x_shift

# Run -------------------------------------------------------------------------------------

def run(network, station, channel, location, stream_index, start_time, end_time, f_min, f_max, Q_min, Q_max, f_points, Q_points, f_frac, log_Q_frac, depth, warmup, sample, devices, cores, diagnostics, signal_step, path):
    numpyro.set_host_device_count(devices)

    t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, f_min, f_max)
    B = f_max - f_min

    initial_plot(path, t, d, f_min, f_max)

    residual = d

    thumbprints = []
    checkpoints = [None] * depth
    signals = jnp.empty((0, 2))
    bounds = jnp.empty((0, 4))

    start_time = perf_counter()
    print("Starting...")

    for iteration in range(depth):
        iteration_start = perf_counter()

        if len(signals) > 0:
            model = get_model(t, d, signals)
            residual = d - model

        grid_start = perf_counter()
        signal = grid_search_orchestrator(path, signals, t, residual, f_min, f_max, Q_min, Q_max, f_points, Q_points, cores)
        grid_elapsed = perf_counter() - grid_start

        signal_bounds = get_bounds(signal, f_min, f_max, Q_min, Q_max, f_frac, log_Q_frac)

        signal = jnp.asarray(signal).reshape(1, 2)
        signal_bounds = jnp.asarray(signal_bounds).reshape(1, 4)

        signals = jnp.concatenate([signals, signal], axis=0)
        bounds = jnp.concatenate([bounds, signal_bounds], axis=0)

        nuts_start = perf_counter()
        if len(signals) % signal_step == 0:
            signals = nuts(t, d, B, signals, bounds, warmup, sample, devices)
        nuts_elapsed = perf_counter() - nuts_start

        if diagnostics == "full":
            checkpoints[iteration] = (signals, bounds)

            log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max)

            if len(signals) % signal_step == 0:
                thumbprints.append((len(signals), glob_ll))

            save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, k2)
            save_signal_diagnostics(path / "diagnostic" / "tables", iteration, fs, Qs, fs_unc, Qs_unc, amplitudes, phases)
            plot_time_series("Time Series", path / "diagnostic" / f"time_series_{len(signals)}.png", t, d, signals)
            plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
            plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc)
            plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "diagnostic" / "covariance" / f"covariance_{len(signals)}.png", signals, covariance)
        else:
            checkpoints[iteration] = (signals, bounds)

            if len(signals) % signal_step == 0:
                log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max)
                thumbprints.append((len(signals), glob_ll))

                save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, k2)
                save_signal_diagnostics(path / "diagnostic" / "tables", iteration, fs, Qs, fs_unc, Qs_unc, amplitudes, phases)
                plot_time_series("Time Series", path / "diagnostic" / f"time_series_{len(signals)}.png", t, d, signals)
                plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
                plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc)
                plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "diagnostic" / "covariance" / f"covariance_{len(signals)}.png", signals, covariance)

        iteration_elapsed = perf_counter() - iteration_start
        total_elapsed = perf_counter() - start_time

        print(f"Signal {len(signals)} completed in {iteration_elapsed:.2f} seconds (total elapsed: {total_elapsed:.2f} seconds)\nGrid search time: {grid_elapsed:.2f} seconds; NUTS time: {nuts_elapsed:.2f} seconds")

    final_start = perf_counter()
    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    checkpoint_by_count = {int(checkpoint[0].shape[0]): checkpoint for checkpoint in checkpoints}

    thumbprints = np.asarray(thumbprints, dtype=float)

    glob_lls = {int(count): float(score) for count, score in thumbprints if np.isfinite(score) and int(count) in checkpoint_by_count}
    refined_checkpoints = {count: checkpoint_by_count[count] for count in glob_lls}

    while True:
        best_count = max(glob_lls, key=glob_lls.get)
        measured_counts = sorted(glob_lls)

        left_counts = [count for count in measured_counts if count < best_count]
        right_counts = [count for count in measured_counts if count > best_count]

        left_limit = left_counts[-1] if left_counts else 1
        right_limit = right_counts[0] if right_counts else depth

        candidates = []

        if best_count - left_limit > 1:
            candidate = (left_limit + best_count) // 2
            if candidate not in glob_lls:
                candidates.append(candidate)

        if right_limit - best_count > 1:
            candidate = (best_count + right_limit) // 2
            if candidate not in glob_lls:
                candidates.append(candidate)

        candidates = sorted(set(candidates))
        if not candidates:
            break

        candidate_refits = nuts_orchestrator([checkpoint_by_count[count] for count in candidates], t, d, B, warmup, sample, devices)

        for count, checkpoint in zip(candidates, candidate_refits):
            refined_checkpoints[count] = checkpoint
            statistics = get_statistics(t, d, B, checkpoint[0], f_min, f_max, Q_min, Q_max)

            log_prob, glob_ll, noise_variance, snr, k2 = statistics[:5]
            glob_lls[count] = float(glob_ll)

            save_iteration_diagnostics(path / "result", checkpoint[0], noise_variance, snr, log_prob, glob_ll, k2)

    best_count = max(glob_lls, key=glob_lls.get)
    signals, bounds = refined_checkpoints[best_count]

    log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max)

    save_signal_diagnostics(path / "result", iteration, fs, Qs, fs_unc, Qs_unc, amplitudes, phases)
    plot_time_series("Time Series", path / "result" / f"time_series_{len(signals)}.png", t, d, signals)
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "result" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
    plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "result" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc)
    plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "result" / "covariance" / f"covariance_{len(signals)}.png", signals, covariance)

    final_elapsed = perf_counter() - final_start

    print(f"Final stage completed in {final_elapsed:.2f} seconds")
    print(f"Dracula completed in {perf_counter() - start_time:.2f} seconds")

    return


if __name__ == "__main__":
    name = "dracula_tohoku_ANMO_01-05mHz_1109_1"
    total_iterations = 6
    f_min_total = 0.00028
    f_max_total = 0.00500
    bandwidth = (f_max_total - f_min_total) / total_iterations

    for i in range(total_iterations):
        print(f"Starting iteration {i}...")
        iteration = i + 1
        f_min = f_min_total + i * bandwidth
        f_max = f_min_total + (i + 1) * bandwidth
        path = Path.cwd() / name / f"dracula_{round(1e3 * f_min, 1)}r{round(1e3 * f_max, 1)}_{iteration}r{total_iterations}"
        path.mkdir(parents=True, exist_ok=True)

        run(network="IU", 
            station="ANMO", 
            channel="LHZ", 
            location="00", 
            stream_index=0, 
            start_time="2011-03-11T19:46:24", 
            end_time="2011-03-25T19:46:24", 

            f_min=f_min, 
            f_max=f_max, 
            Q_min=100,
            Q_max=5000,

            f_points=2000, 
            Q_points=100, 
            f_frac=100, 
            log_Q_frac=3, 
            depth=50, 

            warmup=200, 
            sample=400, 

            devices=7, 
            cores=56, 
            diagnostics="partial", 
            signal_step=10, 

            path=path
        )
