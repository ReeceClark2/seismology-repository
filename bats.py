from typing import Any
import random

from scipy.optimize import minimize as scipy_minimize
from scipy.optimize import Bounds
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.flatten_util import ravel_pytree

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.infer import MCMC, NUTS, init_to_value

import utils


def get_log_prob(t, d, signals):
    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    # G.T = U S Vh
    U, singular_values, _ = jnp.linalg.svd(
        G.T,
        full_matrices=False,
    )

    cutoff = 1e-10 * singular_values[0]
    keep = singular_values > cutoff

    # Project only onto numerically valid directions
    h = U.T @ d
    sum_sq_proj = jnp.sum(jnp.where(keep, h**2, 0.0))

    sum_sq_data = jnp.sum(d**2)
    ratio = sum_sq_proj / jnp.maximum(sum_sq_data, 1e-30)
    ratio = jnp.clip(ratio, 0.0, 1.0 - 1e-12)

    effective_m = jnp.sum(keep)
    N = d.shape[0]

    return 0.5 * (effective_m - N) * jnp.log1p(-ratio)


@jax.jit
def get_model(t, d, signals):
    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    # G.T has shape (N, 2r)
    coefficients = jnp.linalg.lstsq(G.T, d, rcond=None)[0]
    return G.T @ coefficients

def get_noise_variance(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    return (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

def get_snr(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

    return ((m / N) * (1 + ((1 / m) * sum_sq_proj / noise_variance))) ** (1 / 2)


def get_mean_sq_proj(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    return jnp.mean(h ** 2)


def get_glob_ll(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)
    d_scale = jnp.std(d)
    d = d / jnp.maximum(d_scale, jnp.finfo(d.dtype).eps)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    if N <= m + r:
        return -jnp.inf

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    g_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    g_floor = jnp.finfo(g.dtype).eps * g_scale
    eigenvalues = jnp.maximum(eigenvalues, g_floor)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)
    mean_sq_param = (0.5 / m) * jnp.sum(omegas ** 2 + ks ** 2)

    noise_floor = 1

    log_R_delta = jnp.log(jnp.max(jnp.abs(d)) / noise_floor)
    log_R_gamma = jnp.log((jnp.max(d) - jnp.min(d)) / noise_floor)
    log_R_sigma = jnp.log(jnp.std(d) / noise_floor)

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues = jnp.linalg.eigvalsh(b)

    b_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    b_floor = jnp.finfo(b.dtype).eps * b_scale
    eigenvalues = jnp.maximum(eigenvalues, b_floor)

    log_jacobian_factor = -0.5 * jnp.sum(jnp.log(eigenvalues))

    delta_term = (
        (jsp.gammaln(m / 2))
        - jnp.log(2) - log_R_delta
        + (-m / 2) * jnp.log(m * mean_sq_proj / 2.0)
    )

    gamma_term = (
        (jsp.gammaln(r / 2))
        - jnp.log(2) - log_R_gamma
        + (-r / 2) * jnp.log((r * mean_sq_param) / 2)
    )

    sigma_term = (
        (jsp.gammaln((N - m - r) / 2))
        - jnp.log(2) - log_R_sigma
        + ((m + r - N) / 2) * jnp.log(((N * mean_sq_data) - (m * mean_sq_proj)) / 2)
    )

    return delta_term + sigma_term + gamma_term + log_jacobian_factor


def get_amplitudes(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    # G.T has shape (N, 2r)
    coefficients = jnp.linalg.lstsq(G.T, d, rcond=None)[0]

    n_components = fs.shape[0]
    cosine_coefficients = coefficients[:n_components]
    sine_coefficients = coefficients[n_components:]

    amplitudes = jnp.sqrt(
        cosine_coefficients**2 + sine_coefficients**2
    )

    return amplitudes


def get_uncertainties(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    g_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    g_floor = jnp.finfo(g.dtype).eps * g_scale
    eigenvalues = jnp.maximum(eigenvalues, g_floor)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = jnp.linalg.eigh(b)
    b_scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    b_floor = jnp.finfo(b.dtype).eps * b_scale
    eigenvalues = jnp.maximum(eigenvalues, b_floor)
    print(eigenvalues)

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)
    
    signals_uncertainties_flat = jnp.sqrt(noise_variance * jnp.sum(eigenvectors ** 2 / eigenvalues[None, :], axis=1,))
    signals_uncertainties = unravel(signals_uncertainties_flat)

    return signals_uncertainties


def reconcile(
    t,
    d,
    signal_space,
    signals,
    signals_bw,
    nuts_args=None,
    f_tol=1e-5,
):
    if f_tol < 0:
        raise ValueError(f"f_tol must be nonnegative, got {f_tol}")

    fs, ks = utils.unpack_signals(signals)

    fs = np.asarray(fs, dtype=np.float64).reshape(-1)
    ks = np.asarray(ks, dtype=np.float64).reshape(-1)
    signals_bw = jnp.asarray(signals_bw)

    if fs.shape != ks.shape:
        raise ValueError(
            "Frequencies and decay rates must have the same shape; "
            f"got {fs.shape=} and {ks.shape=}."
        )

    if signals_bw.shape[0] != len(fs):
        raise ValueError(
            "signals_bw must have one entry per signal; "
            f"got {signals_bw.shape[0]} bandwidth entries and {len(fs)} signals."
        )

    initial_count = len(fs)

    while len(fs) > 1:
        # Sort signals and signals_bw using the same indices.
        order = np.argsort(fs)

        fs = fs[order]
        ks = ks[order]
        signals_bw = signals_bw[order]

        delta_fs = np.diff(fs)
        close_pairs = np.flatnonzero(delta_fs < f_tol)

        if close_pairs.size == 0:
            break

        # Merge the closest eligible pair.
        pair_index = close_pairs[np.argmin(delta_fs[close_pairs])]
        next_index = pair_index + 1

        merged_f = 0.5 * (fs[pair_index] + fs[next_index])
        merged_k = 0.5 * (ks[pair_index] + ks[next_index])

        # Merge the corresponding bandwidth entries.
        merged_bw = 0.5 * (
            signals_bw[pair_index] + signals_bw[next_index]
        )

        fs = np.concatenate(
            (
                fs[:pair_index],
                np.asarray([merged_f]),
                fs[next_index + 1 :],
            )
        )

        ks = np.concatenate(
            (
                ks[:pair_index],
                np.asarray([merged_k]),
                ks[next_index + 1 :],
            )
        )

        signals_bw = jnp.concatenate(
            (
                signals_bw[:pair_index],
                merged_bw[None, ...],
                signals_bw[next_index + 1 :],
            ),
            axis=0,
        )

        signals = jnp.stack(
            (jnp.asarray(fs), jnp.asarray(ks)),
            axis=-1,
        )

        if nuts_args is not None:
            result = nuts(
                t,
                d,
                signal_space,
                signals,
                signals_bw,
                nuts_kwargs=nuts_args.nuts_kwargs,
                run_kwargs=nuts_args.run_kwargs,
                rng_key_value=nuts_args.rng_key_value,
            )

            if isinstance(result, tuple) and len(result) == 2:
                signals, signals_bw = result
                signals_bw = jnp.asarray(signals_bw)
            else:
                signals = result

            # NUTS may have changed the parameter values.
            fs, ks = utils.unpack_signals(signals)

            fs = np.asarray(fs, dtype=np.float64).reshape(-1)
            ks = np.asarray(ks, dtype=np.float64).reshape(-1)

            if fs.shape != ks.shape:
                raise ValueError(
                    "NUTS returned frequencies and decay rates with "
                    f"different shapes: {fs.shape=} and {ks.shape=}."
                )

            if signals_bw.shape[0] != len(fs):
                raise ValueError(
                    "NUTS returned signals_bw with a different number of "
                    "entries than signals; "
                    f"got {signals_bw.shape[0]} and {len(fs)}."
                )

    # Final synchronized sort.
    order = np.argsort(fs)

    fs = fs[order]
    ks = ks[order]
    signals_bw = signals_bw[order]

    signals = jnp.stack(
        (jnp.asarray(fs), jnp.asarray(ks)),
        axis=-1,
    )

    removed_count = initial_count - len(fs)
    print(
        f"Removed {removed_count} signal"
        f"{'s' if removed_count != 1 else ''}!"
    )

    return signals, signals_bw


def grid_search(t, d, f_points, f_min, f_max, k_points, k_min, k_max, return_probability_surface=False, batch_size=256):
    f_space = jnp.linspace(f_min, f_max, f_points)
    k_space = jnp.geomspace(k_min, k_max, k_points)

    f_grid, k_grid = jnp.meshgrid(
        f_space,
        k_space,
        indexing="ij",
    )

    signal_grid = jnp.stack(
        [f_grid.ravel(), k_grid.ravel()],
        axis=-1,
    )

    evaluate_batch = jax.jit(
        jax.vmap(lambda signal: get_log_prob(t, d, signal))
    )

    results = []

    for start in range(0, signal_grid.shape[0], batch_size):
        batch = signal_grid[start:start + batch_size]
        results.append(evaluate_batch(batch))

    log_probs = jnp.concatenate(results)

    log_prob_space = log_probs.reshape(
        f_space.size,
        k_space.size,
    )

    signal = utils.get_best_signal(
        (f_space, k_space, log_prob_space)
    )

    if return_probability_surface:
        return signal, (f_space, k_space, log_prob_space)

    return signal


def bats_model(
    t: jax.Array,
    d: jax.Array,
    f_loc: jax.Array,
    f_scale: float | jax.Array,
    k_loc: jax.Array,
    k_scale: float | jax.Array,
    f_min: float,
    f_max: float,
    k_min: float,
    k_max: float,
) -> None:
    f_loc = jnp.atleast_1d(jnp.asarray(f_loc))
    k_loc = jnp.atleast_1d(jnp.asarray(k_loc))

    f_scale = jnp.broadcast_to(
        jnp.asarray(f_scale, dtype=f_loc.dtype),
        f_loc.shape,
    )

    k_scale = jnp.broadcast_to(
        jnp.asarray(k_scale, dtype=k_loc.dtype),
        k_loc.shape,
    )

    # Bounds in ordinary frequency space
    f_min = jnp.asarray(f_min, dtype=f_loc.dtype)
    f_max = jnp.asarray(f_max, dtype=f_loc.dtype)

    # k must be positive before converting to log space
    log_k_min = jnp.log(
        jnp.asarray(k_min, dtype=k_loc.dtype)
    )
    log_k_max = jnp.log(
        jnp.asarray(k_max, dtype=k_loc.dtype)
    )

    log_k_loc = jnp.log(k_loc)

    # Local rectangular region in f space
    f_low = jnp.maximum(
        f_loc - f_scale,
        f_min,
    )
    f_high = jnp.minimum(
        f_loc + f_scale,
        f_max,
    )

    # Local rectangular region in log(k) space
    log_k_low = jnp.maximum(
        log_k_loc - k_scale,
        log_k_min,
    )
    log_k_high = jnp.minimum(
        log_k_loc + k_scale,
        log_k_max,
    )

    # Sample independently from the clipped rectangle
    fs = numpyro.sample(
        "fs",
        dist.Uniform(f_low, f_high).to_event(1),
    )

    log_ks = numpyro.sample(
        "log_ks",
        dist.Uniform(log_k_low, log_k_high).to_event(1),
    )

    ks = numpyro.deterministic(
        "ks",
        jnp.exp(log_ks),
    )

    signals = jnp.stack(
        (fs, ks),
        axis=-1,
    )

    numpyro.factor(
        "surface",
        get_log_prob(t, d, signals),
    )


def nuts(
    t,
    d,
    signal_space,
    signals,
    signals_bw,
    nuts_kwargs,
    mcmc_kwargs,
    run_kwargs,
    rng_key_value,
):
    f_init, k_init = utils.unpack_signals(signals)

    f_min = signal_space.f_min
    f_max = signal_space.f_max
    k_min = signal_space.k_min
    k_max = signal_space.k_max

    f_init = jnp.atleast_1d(jnp.asarray(f_init))
    k_init = jnp.atleast_1d(jnp.asarray(k_init))

    f_bw, k_bw = utils.unpack_signals(signals_bw)

    f_bw = jnp.broadcast_to(
        jnp.asarray(f_bw, dtype=f_init.dtype),
        f_init.shape,
    )
    k_bw = jnp.broadcast_to(
        jnp.asarray(k_bw, dtype=k_init.dtype),
        k_init.shape,
    )

    log_k_init = jnp.log(k_init)

    f_low = jnp.maximum(f_init - f_bw, f_min)
    f_high = jnp.minimum(f_init + f_bw, f_max)

    log_k_min = jnp.log(k_min)
    log_k_max = jnp.log(k_max)

    log_k_low = jnp.maximum(log_k_init - k_bw, log_k_min)
    log_k_high = jnp.minimum(log_k_init + k_bw, log_k_max)

    if bool(jnp.any(f_low >= f_high)):
        raise ValueError(
            f"Empty frequency interval: "
            f"low={f_low}, high={f_high}"
        )

    if bool(jnp.any(log_k_low >= log_k_high)):
        raise ValueError(
            f"Empty log-k interval: "
            f"low={log_k_low}, high={log_k_high}"
        )

    # Keep initial values away from hard Uniform boundaries.
    f_width = f_high - f_low
    log_k_width = log_k_high - log_k_low

    f_eps = 1e-6 * jnp.maximum(f_width, 1.0)
    log_k_eps = 1e-6 * jnp.maximum(log_k_width, 1.0)

    f_init_safe = jnp.clip(
        f_init,
        f_low + f_eps,
        f_high - f_eps,
    )

    log_k_init_safe = jnp.clip(
        log_k_init,
        log_k_low + log_k_eps,
        log_k_high - log_k_eps,
    )

    signals_init = jnp.stack(
        (f_init_safe, jnp.exp(log_k_init_safe)),
        axis=-1,
    )

    surface_value = get_log_prob(t, d, signals_init)

    if not bool(jnp.isfinite(surface_value)):
        raise ValueError(
            f"Initial surface log probability is not finite: "
            f"{surface_value}"
        )

    init_strategy = init_to_value(
        values={
            "fs": f_init_safe,
            "log_ks": log_k_init_safe,
        }
    )

    nuts_config = {
        "init_strategy": init_strategy,
    }
    nuts_config.update(nuts_kwargs)

    if rng_key_value is None:
        rng_key_value = random.randint(1, 1_000)

    kernel = NUTS(
        bats_model,
        **nuts_config,
    )

    mcmc = MCMC(
        kernel,
        **mcmc_kwargs,
    )

    mcmc.run(
        jax.random.PRNGKey(int(rng_key_value)),
        t=t,
        d=d,
        f_loc=f_init,
        f_scale=f_bw,
        k_loc=k_init,
        k_scale=k_bw,
        f_min=f_min,
        f_max=f_max,
        k_min=k_min,
        k_max=k_max,
        extra_fields=("potential_energy",),
        **run_kwargs,
    )

    samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)

    potential_energy = extra_fields["potential_energy"]

    best_flat_index = jnp.argmin(potential_energy.ravel())

    chain_index, draw_index = jnp.unravel_index(
        best_flat_index,
        potential_energy.shape,
    )

    best_fs = samples["fs"][chain_index, draw_index]
    best_ks = samples["ks"][chain_index, draw_index]

    return jnp.stack(
        (best_fs, best_ks),
        axis=-1,
    )


def minimize(
    t,
    d,
    signal_space,
    signals,
    maxiter=2_000,
):
    signals = jnp.asarray(signals)

    f_min = float(signal_space.f_min)
    f_max = float(signal_space.f_max)
    k_min = float(signal_space.k_min)
    k_max = float(signal_space.k_max)

    if not f_max > f_min:
        raise ValueError(f"Expected f_max > f_min, got {f_min=} and {f_max=}")

    if not k_max > k_min:
        raise ValueError(f"Expected k_max > k_min, got {k_min=} and {k_max=}")

    original_shape = signals.shape

    # Physical parameters:
    # [f_0, k_0, f_1, k_1, ...]
    x0_physical = np.asarray(signals, dtype=np.float64).reshape(-1)

    if len(x0_physical) % 2 != 0:
        raise ValueError(
            "Expected an even number of parameters arranged as "
            "[frequency, decay_rate, ...]"
        )

    lower = np.tile(
        np.asarray([f_min, k_min], dtype=np.float64),
        len(x0_physical) // 2,
    )
    upper = np.tile(
        np.asarray([f_max, k_max], dtype=np.float64),
        len(x0_physical) // 2,
    )

    widths = upper - lower

    if np.any(x0_physical < lower) or np.any(x0_physical > upper):
        raise ValueError(
            "Initial parameters are outside the specified bounds.\n"
            f"x0={x0_physical}\n"
            f"lower={lower}\n"
            f"upper={upper}"
        )

    # Normalize physical parameters to [0, 1].
    x0_normalized = (x0_physical - lower) / widths

    lower_jax = jnp.asarray(lower)
    widths_jax = jnp.asarray(widths)

    def normalized_to_physical(x_normalized):
        return lower_jax + widths_jax * x_normalized

    def objective_normalized(x_normalized):
        x_physical = normalized_to_physical(x_normalized)
        signals_current = x_physical.reshape(original_shape)

        # This is the objective being minimized.
        return -get_log_prob(t, d, signals_current)

    # Compute value, gradient, and exact Hessian with JAX.
    objective_derivatives = jax.jit(
        jax.value_and_grad(objective_normalized)
    )
    hessian_normalized = jax.jit(
        jax.hessian(objective_normalized)
    )

    def objective_and_gradient(x_normalized):
        x_normalized = np.asarray(x_normalized, dtype=np.float64)

        value, gradient = objective_derivatives(
            jnp.asarray(x_normalized)
        )

        value = float(value)
        gradient = np.asarray(gradient, dtype=np.float64).reshape(-1)

        if not np.isfinite(value):
            raise FloatingPointError(
                f"Non-finite objective at normalized parameters "
                f"{x_normalized}"
            )

        if not np.all(np.isfinite(gradient)):
            raise FloatingPointError(
                f"Non-finite gradient at normalized parameters "
                f"{x_normalized}"
            )

        return value, gradient

    def hessian_callback(x_normalized):
        x_normalized = np.asarray(x_normalized, dtype=np.float64)

        hessian = np.asarray(
            hessian_normalized(jnp.asarray(x_normalized)),
            dtype=np.float64,
        )

        if not np.all(np.isfinite(hessian)):
            raise FloatingPointError(
                f"Non-finite Hessian at normalized parameters "
                f"{x_normalized}"
            )

        # Numerical roundoff can make an analytically symmetric Hessian
        # slightly nonsymmetric.
        return 0.5 * (hessian + hessian.T)

    # Every normalized parameter has bounds [0, 1].
    bounds = Bounds(
        lb=np.zeros_like(x0_normalized),
        ub=np.ones_like(x0_normalized),
    )

    result = scipy_minimize(
        objective_and_gradient,
        x0_normalized,
        method="trust-constr",
        jac=True,
        hess=hessian_callback,
        bounds=bounds,
        options={
            "maxiter": maxiter,
            "gtol": 1e-8,
            "xtol": 1e-12,
            "verbose": 0,
        },
    )

    if not np.all(np.isfinite(result.x)):
        raise FloatingPointError(
            "trust-constr returned non-finite normalized parameters"
        )

    result_physical = lower + result.x * widths

    if not np.all(np.isfinite(result_physical)):
        raise FloatingPointError(
            "trust-constr returned non-finite physical parameters"
        )

    if not result.success:
        print(f"trust-constr warning: {result.message}")

    return jnp.asarray(result_physical).reshape(original_shape)




#--------------------- Testing -----------------------------------

def get_b_eigendecomposition(t, d, signals):
    """
    Return the raw eigendecomposition of the curvature matrix b.

    The parameter ordering is assumed to match signals.reshape(-1), e.g.

        [f_0, k_0, f_1, k_1, ...]

    """
    signals = jnp.asarray(signals)

    r = signals.shape[0]
    m = signals.size

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    # Same definition as in get_uncertainties.
    b = (-m / 2.0) * jax.hessian(objective)(theta)

    # Remove small asymmetries caused by numerical differentiation.
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = jnp.linalg.eigh(b)

    return b, eigenvalues, eigenvectors


def reconcile_noise_floor(
    t,
    d,
    signal_space,
    signals,
    signals_bw,
    nuts_args=None,
    floor_factor=1.0,
    min_signals=1,
):
    """
    Remove signals associated with eigenmodes of b that hit the
    numerical eigenvalue floor.

    Parameters
    ----------
    t, d :
        Time samples and data values.

    signal_space :
        Passed through to nuts().

    signals :
        Array with shape (n_signals, 2), containing frequency and
        decay-rate parameters.

    signals_bw :
        Array whose first axis indexes signals.

    nuts_args :
        Optional NUTS configuration. If provided, NUTS is rerun after
        every signal removal.

    floor_factor :
        Multiplier for the numerical eigenvalue floor. A value of 1.0
        detects eigenvalues at or below eps * max_eigenvalue. A larger
        value detects moderately small eigenvalues as well.

    min_signals :
        Do not remove signals below this number.

    Returns
    -------
    signals, signals_bw
        The reconciled signal parameters and bandwidths.
    """
    if floor_factor < 1:
        raise ValueError(
            f"floor_factor must be at least 1, got {floor_factor}"
        )

    if min_signals < 0:
        raise ValueError(
            f"min_signals must be nonnegative, got {min_signals}"
        )

    fs, ks = utils.unpack_signals(signals)

    fs = np.asarray(fs, dtype=np.float64).reshape(-1)
    ks = np.asarray(ks, dtype=np.float64).reshape(-1)
    signals_bw = jnp.asarray(signals_bw)

    if fs.shape != ks.shape:
        raise ValueError(
            "Frequencies and decay rates must have the same shape; "
            f"got {fs.shape=} and {ks.shape=}."
        )

    if signals_bw.shape[0] != len(fs):
        raise ValueError(
            "signals_bw must have one entry per signal; "
            f"got {signals_bw.shape[0]} bandwidth entries and "
            f"{len(fs)} signals."
        )

    initial_count = len(fs)

    while len(fs) > min_signals:
        signals = jnp.stack(
            (jnp.asarray(fs), jnp.asarray(ks)),
            axis=-1,
        )

        b, raw_eigenvalues, eigenvectors = get_b_eigendecomposition(
            t,
            d,
            signals,
        )

        # Numerical floor used by the uncertainty calculation.
        b_scale = jnp.maximum(
            jnp.max(jnp.abs(raw_eigenvalues)),
            1.0,
        )

        eigenvalue_floor = (
            floor_factor
            * jnp.finfo(b.dtype).eps
            * b_scale
        )

        # Modes at or below the numerical floor are considered
        # unconstrained or degenerate.
        bad_modes = raw_eigenvalues <= eigenvalue_floor

        if not bool(jnp.any(bad_modes)):
            break

        bad_mode_indices = jnp.flatnonzero(bad_modes)

        # Each column of eigenvectors is an eigenmode. Compute the
        # contribution of each parameter to all bad modes.
        parameter_scores = jnp.sum(
            eigenvectors[:, bad_mode_indices] ** 2,
            axis=1,
        )

        # The flattened parameter order is assumed to be:
        #
        #   [f_0, k_0, f_1, k_1, ...]
        #
        # Combine the frequency and decay-rate scores for each signal.
        signal_scores = parameter_scores.reshape(len(fs), 2).sum(axis=1)

        remove_index = int(jnp.argmax(signal_scores))

        print(
            "Removing signal "
            f"{remove_index}: "
            f"f={fs[remove_index]:.6g}, "
            f"k={ks[remove_index]:.6g}; "
            f"bad_modes={int(jnp.sum(bad_modes))}, "
            f"score={float(signal_scores[remove_index]):.6g}"
        )

        # Remove the same signal index from all signal-related arrays.
        keep = np.ones(len(fs), dtype=bool)
        keep[remove_index] = False

        fs = fs[keep]
        ks = ks[keep]
        signals_bw = signals_bw[keep]

        signals = jnp.stack(
            (jnp.asarray(fs), jnp.asarray(ks)),
            axis=-1,
        )

        # Optionally rerun NUTS after removing a degenerate signal.
        if nuts_args is not None:
            result = nuts(
                t,
                d,
                signal_space,
                signals,
                signals_bw,
                nuts_kwargs=nuts_args.nuts_kwargs,
                run_kwargs=nuts_args.run_kwargs,
                rng_key_value=nuts_args.rng_key_value,
            )

            if isinstance(result, tuple) and len(result) == 2:
                signals, signals_bw = result
                signals_bw = jnp.asarray(signals_bw)
            else:
                signals = result

            fs, ks = utils.unpack_signals(signals)

            fs = np.asarray(fs, dtype=np.float64).reshape(-1)
            ks = np.asarray(ks, dtype=np.float64).reshape(-1)

            if fs.shape != ks.shape:
                raise ValueError(
                    "NUTS returned frequencies and decay rates with "
                    f"different shapes: {fs.shape=} and {ks.shape=}."
                )

            if signals_bw.shape[0] != len(fs):
                raise ValueError(
                    "NUTS returned signals_bw with a different number "
                    "of entries than signals; "
                    f"got {signals_bw.shape[0]} and {len(fs)}."
                )

    # Final synchronized sorting.
    order = np.argsort(fs)

    fs = fs[order]
    ks = ks[order]
    signals_bw = signals_bw[order]

    signals = jnp.stack(
        (jnp.asarray(fs), jnp.asarray(ks)),
        axis=-1,
    )

    removed_count = initial_count - len(fs)

    print(
        f"Removed {removed_count} signal"
        f"{'s' if removed_count != 1 else ''}!"
    )

    return signals, signals_bw