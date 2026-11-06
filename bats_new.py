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
import arviz as az

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
    k_min: float
    k_max: float
    f_points: int
    k_points: int

# Data Acquisition -----------------------------------------------------------------------

def get_observed_data(network, station, channel, location, stream_index, start_time, end_time):
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

    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

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
    
    fs, ks = unpack_signals(signals)

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

def grid_search(t, d, f_min, f_max, k_min, k_max, f_points, k_points, batch_size=512):
    f_space = jnp.linspace(f_min, f_max, f_points)
    B = f_max - f_min

    k_margin = 1e-6 * (k_max - k_min)
    k_space = jnp.geomspace(k_min + k_margin, k_max - k_margin, k_points)

    f_grid, k_grid = jnp.meshgrid(f_space, k_space, indexing="ij")
    signal_grid = jnp.stack([f_grid.ravel(), k_grid.ravel()], axis=-1)

    evaluate_batch = jax.jit(
        jax.vmap(lambda signal: get_log_prob(t, d, B, signal))
    )

    results = []
    
    for start in range(0, signal_grid.shape[0], batch_size):
        batch = signal_grid[start:start + batch_size]
        results.append(evaluate_batch(batch))

    log_probs = jnp.concatenate(results)
    log_prob_space = log_probs.reshape(f_space.size, k_space.size)

    probability_surface = (f_space, k_space, log_prob_space)
    
    return probability_surface

def objective(t, d, B, f_min, f_max, k_min, k_max):
    fs = numpyro.sample(
        "fs",
        numpyro.distributions.Uniform(f_min, f_max).to_event(1),
    )

    log_k_min = jnp.log(k_min)
    log_k_max = jnp.log(k_max)

    log_ks = numpyro.sample(
        "log_ks",
        numpyro.distributions.Uniform(log_k_min, log_k_max).to_event(1),
    )

    ks = numpyro.deterministic(
        "ks",
        jnp.exp(log_ks),
    )

    signals = jnp.stack(
        (fs, ks),
        axis=-1,
    )

    numpyro.factor("surface", get_log_prob(t, d, B, signals))
 
def nuts(t, d, B, signals, bounds, warmup, sample, chains, rng_key_value=None):
    f_init, k_init = unpack_signals(signals, sort=False)

    bounds = jnp.asarray(bounds).reshape(-1, 4)

    fs_min = bounds[:, 0].astype(f_init.dtype)
    fs_max = bounds[:, 1].astype(f_init.dtype)
    ks_min = bounds[:, 2].astype(k_init.dtype)
    ks_max = bounds[:, 3].astype(k_init.dtype)

    log_k_init = jnp.log(k_init)

    init_strategy = init_to_value(values={"fs": f_init, "log_ks": log_k_init})

    if rng_key_value is None:
        rng_key_value = random.randint(1, 10_000)

    rng_key = jax.random.PRNGKey(int(rng_key_value))
    warmup_key, sample_key = jax.random.split(rng_key)

    kernel = NUTS(objective, init_strategy=init_strategy, max_tree_depth=5)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=sample, num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.warmup(warmup_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, k_min=ks_min, k_max=ks_max, extra_fields=("potential_energy",))

    warmup_samples = mcmc.get_samples(group_by_chain=True)
    warmup_extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    mcmc.run(mcmc.post_warmup_state.rng_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, k_min=ks_min, k_max=ks_max, extra_fields=("potential_energy",))

    sample_samples = mcmc.get_samples(group_by_chain=True)
    sample_extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    samples = {name: jnp.concatenate([warmup_samples[name], sample_samples[name]], axis=1) for name in sample_samples}
    extra_fields = {name: jnp.concatenate([warmup_extra_fields[name], sample_extra_fields[name]], axis=1) for name in sample_extra_fields}

    potential_energy = extra_fields["potential_energy"]

    best_flat_index = jnp.argmin(potential_energy.ravel())
    chain_index, draw_index = jnp.unravel_index(best_flat_index, potential_energy.shape)

    best_fs = samples["fs"][chain_index, draw_index]
    best_ks = samples["ks"][chain_index, draw_index]

    signals = jnp.stack((best_fs, best_ks), axis=-1)
    return signals

# Orchestrators ---------------------------------------------------------------------------

def _run_grid_search(args):
    t, d, task = args

    return grid_search(
        t=t,
        d=d,
        f_min=task.f_min,
        f_max=task.f_max,
        k_min=task.k_min,
        k_max=task.k_max,
        f_points=task.f_points,
        k_points=task.k_points,
    )

def grid_search_orchestrator(signals, t, d, f_min, f_max, k_min, k_max, f_points, k_points, cores):
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
                k_min=k_min,
                k_max=k_max,
                f_points=stop - start,
                k_points=k_points,  # Do not divide this
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
    k_space = surfaces[0][1]

    probability_surface = (
        f_space,
        k_space,
        log_prob_space,
    )

    plot_probability_surface(f"Probability Surface {len(signals)}", path / "diagnostic" / "probability_surface" / f"probability_surface_{len(signals)}.png", probability_surface)

    f_space, k_space, log_prob_space = probability_surface
    best_index = jnp.unravel_index(jnp.argmax(log_prob_space), log_prob_space.shape)

    f_index, k_index = best_index
    f_space[f_index]
    signal = (f_space[f_index], k_space[k_index])

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

def nuts_orchestrator(checkpoints, t, d, B, warmup, sample, cores):
    if not checkpoints:
        return []

    if cores < 1:
        raise ValueError("cores must be at least 1")

    n_checkpoints = len(checkpoints)
    n_workers = min(cores, n_checkpoints)
    cores_per_worker = max(1, cores // n_checkpoints)

    tasks = [(checkpoint, t, d, B, warmup, sample, cores_per_worker, index) for index, checkpoint in enumerate(checkpoints)]
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        optimized_checkpoints = list(executor.map(_nuts_worker, tasks))

    return optimized_checkpoints

# Evaluation (critical) -------------------------------------------------------------------

def get_mean_sq_proj(t, d, B, signals):
    fs, ks = unpack_signals(signals)

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

def get_glob_ll(t, d, signals, f_min, f_max, k_min, k_max, ratio_delta=1e2, ratio_gamma=1e2, ratio_sigma=1e2):
    fs, ks = unpack_signals(signals)

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
    omega_scale = 2 * jnp.pi * (f_max - f_min) / 2
    k_scale = (k_max - k_min) / 2

    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)
    mean_sq_param = (1 / r) * jnp.sum((omegas / omega_scale) ** 2 + (ks / k_scale) ** 2)

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
    fs, ks = unpack_signals(signals)

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
    fs, ks = unpack_signals(signals)

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
    fs, ks = unpack_signals(signals)

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
    fs, ks = unpack_signals(signals)

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
    fs, ks = unpack_signals(signals)
    
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
    fs, ks = unpack_signals(signals)

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
    ks_unc = variances[1::2]

    return fs_unc, ks_unc

# Evaluation (all) ------------------------------------------------------------------------

def get_statistics(t, d, B, signals, f_min, f_max, k_min, k_max, ratio_delta=1e2, ratio_gamma=1e2, ratio_sigma=1e2):
    fs, ks = unpack_signals(signals)

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

    omega_scale = 2 * jnp.pi * (f_max - f_min) / 2
    k_scale = (k_max - k_min) / 2

    mean_sq_param = (1 / r) * jnp.sum((omegas / omega_scale) ** 2 + (ks / k_scale) ** 2)

    log_R_delta = jnp.log(ratio_delta)
    log_R_gamma = jnp.log(ratio_gamma)
    log_R_sigma = jnp.log(ratio_sigma)

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, B, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    b_eigenvalues, b_eigenvectors = jnp.linalg.eigh(b)

    unc = jnp.sqrt(noise_variance * jnp.sum(b_eigenvectors ** 2 / b_eigenvalues[None, :], axis=1))
    fs_unc = unc[::2]
    ks_unc = unc[1::2]

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

    return log_prob, glob_ll, noise_variance, snr, k2, fs, ks, fs_unc, ks_unc, amplitudes, phases

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

def save_signal_diagnostics(path, iteration, fs, ks, fs_unc, ks_unc, amplitudes, phases):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostic_path / f"signals_iteration_{iteration + 1}.csv"

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency", "Frequency Uncertainty", "Decay Rate", "Decay Rate Uncertainty", "Amplitude", "Phase"])
        writer.writerows(zip(fs, fs_unc, ks, ks_unc, amplitudes, phases))

def plot_probability_surface(title, path, probability_surface):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f_space, k_space, log_prob_space = probability_surface
    log_prob_space = log_prob_space.T

    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap = ax.pcolormesh(f_space, k_space, log_prob_space, shading="auto", cmap="Grays")

    ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Log Probability")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

def plot_signal_space(title, path, fs, ks, fs_unc, ks_unc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = np.asarray(fs)
    ks = np.asarray(ks)
    fs_unc = np.asarray(fs_unc)
    ks_unc = np.asarray(ks_unc)

    Q = np.pi * fs / ks
    Q_unc = np.sqrt((np.pi * fs_unc / ks) ** 2 + (np.pi * fs * ks_unc / ks**2) ** 2)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(fs, Q, color="red", s=15, zorder=3, label="Signals")

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    for f, q, f_unc, q_unc in zip(fs, Q, fs_unc, Q_unc):
        for sigma, alpha in ((3.0, 0.06), (2.0, 0.12), (1.0, 0.22)):
            ellipse = Ellipse(xy=(f, q), width=2.0 * sigma * f_unc, height=2.0 * sigma * q_unc, facecolor="red", edgecolor="red", alpha=alpha, linewidth=0.8)
            ax.add_patch(ellipse)

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
        model = get_model(t, d, signals, domain="time")

        ax.plot(t, model, color="red", label="Model", alpha=0.6)
        ax.plot(t, d-model, color="blue", label="Residual", alpha=0.2)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

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
        fs, ks = unpack_signals(signals)
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
    ks = signals[:, 1]

    return fs, ks

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

def get_bounds(signals, f_min, f_max, k_min, k_max, f_fraction=1 / 8, log_k_fraction=1 / 8):
    fs, ks = unpack_signals(signals)

    frequency_halfwidth = (f_max - f_min) / (2 * f_fraction)

    log_k_space_min = jnp.log(k_min)
    log_k_space_max = jnp.log(k_max)
    log_k_halfwidth = (log_k_space_max - log_k_space_min) / (2 * log_k_fraction)

    frequency_min = jnp.maximum(fs - frequency_halfwidth, f_min)
    frequency_max = jnp.minimum(fs + frequency_halfwidth, f_max)

    log_ks = jnp.log(ks)
    log_k_min = jnp.maximum(log_ks - log_k_halfwidth, log_k_space_min)
    log_k_max = jnp.minimum(log_ks + log_k_halfwidth, log_k_space_max)

    decay_rate_min = jnp.exp(log_k_min)
    decay_rate_max = jnp.exp(log_k_max)

    return jnp.stack((frequency_min, frequency_max, decay_rate_min, decay_rate_max), axis=-1)

def get_shift(x, y):
    a, b, c = np.polyfit(x, y, 2)
    x_shift = -b / (2 * a)

    return x_shift

# Run -------------------------------------------------------------------------------------

def run(network, station, channel, location, stream_index, start_time, end_time, f_min, f_max, k_min, k_max, f_points, k_points, f_frac, log_k_frac, depth, warmup, sample, devices, cores, diagnostics, signal_step, path):
    numpyro.set_host_device_count(devices)

    t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time)
    B = f_max - f_min

    initial_plot(path, t, d, f_min, f_max)

    residual = d

    thumbprints = []
    checkpoints = [jnp.empty((i,)) for i in range(1, depth + 1)]
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
        signal = grid_search_orchestrator(signals, t, residual, f_min, f_max, k_min, k_max, f_points, k_points, cores)
        grid_elapsed = perf_counter() - grid_start

        signal_bounds = get_bounds(signal, f_min, f_max, k_min, k_max, f_frac, log_k_frac)

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

            log_prob, glob_ll, noise_variance, snr, k2, fs, ks, fs_unc, ks_unc, amplitudes, phases = get_statistics(t, d, B, signals, f_min, f_max, k_min, k_max)

            if len(signals) % signal_step == 0:
                thumbprints.append((len(signals), glob_ll))

            save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, k2)
            save_signal_diagnostics(path / "diagnostic" / "tables", iteration, fs, ks, fs_unc, ks_unc, amplitudes, phases)
            plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
            plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, ks, fs_unc, ks_unc)
        else:
            if len(signals) % signal_step == 0:
                checkpoints[iteration] = (signals, bounds)
                log_prob, glob_ll, noise_variance, snr, k2, fs, ks, fs_unc, ks_unc, amplitudes, phases = get_statistics(t, d, B, signals, f_min, f_max, k_min, k_max)
                thumbprints.append((len(signals), glob_ll))

                save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, k2)
                save_signal_diagnostics(path / "diagnostic" / "tables", iteration, fs, ks, fs_unc, ks_unc, amplitudes, phases)
                plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
                plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, ks, fs_unc, ks_unc)

        iteration_elapsed = perf_counter() - iteration_start
        total_elapsed = perf_counter() - start_time

        print(f"Signal {len(signals)} completed in {iteration_elapsed:.2f} seconds (total elapsed: {total_elapsed:.2f} seconds)\nGrid search time: {grid_elapsed:.2f} seconds; NUTS time: {nuts_elapsed:.2f} seconds")

    final_start = perf_counter()
    thumbprints = jnp.asarray(thumbprints)
    best_signal = int(get_shift(thumbprints[:, 0], thumbprints[:, 1]))

    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint is not None]
    best_checkpoint_index = min(range(len(checkpoints)), key=lambda i: abs(len(checkpoints[i][0]) - best_signal))
    start = max(0, best_checkpoint_index - 2)
    stop = min(len(checkpoints), best_checkpoint_index + 3)

    checkpoints = nuts_orchestrator(checkpoints[start:stop], t, d, B, 500, 1_000, devices)

    glob_lls = jnp.zeros(len(checkpoints))

    for ind, (signals, bounds) in enumerate(checkpoints):
        log_prob, glob_ll, noise_variance, snr, k2, fs, ks, fs_unc, ks_unc, amplitudes, phases = get_statistics(t, d, B, signals, f_min, f_max, k_min, k_max)
        save_iteration_diagnostics(path / "result", signals, noise_variance, snr, log_prob, glob_ll, k2)
        glob_lls = glob_lls.at[ind].set(glob_ll)

    best_index = jnp.argmax(glob_lls)
    signals, bounds = checkpoints[best_index]

    log_prob, glob_ll, noise_variance, snr, k2, fs, ks, fs_unc, ks_unc, amplitudes, phases = get_statistics(t, d, B, signals, f_min, f_max, k_min, k_max)

    save_signal_diagnostics(path / "result", iteration, fs, ks, fs_unc, ks_unc, amplitudes, phases)
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "result" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals)
    plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "result" / f"signal_space_{len(signals)}.png", fs, ks, fs_unc, ks_unc)
    final_elapsed = perf_counter() - final_start

    print(f"Final stage completed in {final_elapsed:.2f} seconds")
    
    print(f"Dracula completed in {perf_counter() - start_time:.2f} seconds")

    return


if __name__ == "__main__":
    path = Path.cwd() / "dracula_1105_2"
    path.mkdir(parents=True, exist_ok=True)

    run(
        network="IU",
        station="ANMO",
        channel="LHZ",
        location="00",
        stream_index=0,
        start_time="2025-07-30T04:24:52",
        end_time="2025-08-10T17:24:52",

        f_min=0.00028,
        f_max=0.00150,
        k_min=4.0e-7,
        k_max=5.0e-5,

        f_points=2000,
        k_points=100,

        f_frac=100,
        log_k_frac=3,

        depth=60,

        warmup=200,
        sample=400,
        devices=10,
        cores=40,

        diagnostics="partial",
        signal_step=10,

        path=path
    )
