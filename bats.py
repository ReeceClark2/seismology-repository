from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.flatten_util import ravel_pytree

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.infer import MCMC, NUTS, init_to_value

import utils


def get_log_prob(t: jax.Array, d: jax.Array, signals) -> jax.Array:
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

    ratio = sum_sq_proj / jnp.maximum(sum_sq_data, 1e-30)
    ratio = jnp.clip(ratio, 0.0, 1.0 - 1e-12)

    return 0.5 * (m - N) * jnp.log1p(-ratio)

@jax.jit
def get_model(t: jax.Array, d: jax.Array, signals) -> jax.Array:
    fs, ks = utils.unpack_signals(signals)

    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-19)

    # Bretthorst Eq. 3.6: orthonormal functions H
    T = (eigenvectors / jnp.sqrt(eigenvalues)).T
    
    H = T @ G
    h = H @ d
    model = h @ H

    # Transform orthogonal amplitudes (h) back to physical amplitudes (A)
    B = h @ T

    return model

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
    scale = 1 / min(d)
    d *= scale

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

    eigenvalues = jnp.linalg.eigvalsh(b)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

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


def grid_search(t, d, f_points, f_min, f_max, k_points, k_min, k_max, return_probability_surface=False):
    f_space = jnp.asarray(jnp.linspace(f_min, f_max, f_points))
    k_space = jnp.asarray(jnp.geomspace(k_min, k_max, k_points))
    
    f_grid, k_grid = jnp.meshgrid(f_space, k_space, indexing="ij")

    signals = jnp.stack(
        [f_grid.ravel(), k_grid.ravel()],
        axis=-1
    )

    log_probs = jax.vmap(
        lambda signal: get_log_prob(t, d, signal)
    )(signals)

    log_prob_space = log_probs.reshape(
        f_space.size,
        k_space.size
    )

    signal = utils.get_best_signal((f_space, k_space, log_prob_space))
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

    # Frequency bounds
    f_low = f_loc - f_scale
    f_high = f_loc + f_scale

    fs = numpyro.sample(
        "fs",
        dist.Uniform(f_low, f_high).to_event(1),
    )

    # Log-decay-rate bounds
    k_floor = jnp.asarray(1e-12, dtype=k_loc.dtype)
    log_k_loc = jnp.log(jnp.maximum(k_loc, k_floor))

    log_k_low = log_k_loc - k_scale
    log_k_high = log_k_loc + k_scale

    # Sample the bounded variable directly in log space
    log_ks = numpyro.sample(
        "log_ks",
        dist.Uniform(log_k_low, log_k_high).to_event(1),
    )

    # Transform back to ordinary decay-rate space
    ks = numpyro.deterministic(
        "ks",
        jnp.exp(log_ks),
    )

    signals = jnp.stack((fs, ks), axis=-1)

    numpyro.factor(
        "surface",
        get_log_prob(t, d, signals),
    )


def nuts(t, d, signals, signals_bw, nuts_kwargs, mcmc_kwargs, run_kwargs):
    f_init, k_init = utils.unpack_signals(signals)
    f_bw, k_bw = utils.unpack_signals(signals_bw)

    init_strategy = init_to_value(values={"fs": f_init, "ks": k_init})

    nuts_config: dict[str, Any] = {
        "init_strategy": init_strategy,
    }
    nuts_config.update(nuts_kwargs)

    kernel = NUTS(bats_model, **nuts_config)
    mcmc = MCMC(kernel, **mcmc_kwargs)
    mcmc.run(
        jax.random.PRNGKey(int(42)),
        t,
        d,
        f_init,
        f_bw,
        k_init,
        k_bw,
        **run_kwargs,
        extra_fields=("potential_energy",)
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

    best_signals = jnp.stack(
        (jnp.asarray(best_fs), jnp.asarray(best_ks)),
        axis=-1,
    )

    return best_signals
    