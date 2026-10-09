# Internal Python libraries
import random
from typing import Any

# External libraries
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.flatten_util import ravel_pytree
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import Bounds
from scipy.optimize import minimize as scipy_minimize

# Repository files
import utils


jax.config.update("jax_enable_x64", True)


def get_log_prob(
        t: jax.Array, 
        d: jax.Array,
        signals
    ):
    '''
    Bretthorst Eq. 3.17
    '''
    
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
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    # Bretthorst Eq. 3.5: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    N = d.shape[0]

    ratio = sum_sq_proj / sum_sq_data

    return 0.5 * (m - N) * jnp.log1p(-ratio)

def get_gram(
        t: jax.Array, 
        d: jax.Array,
        signals
    ):
    '''
    Bretthorst Eq. 3.4
    '''

    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    g = G @ G.T
    g = 0.5 * (g + g.T)

    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    return g, eigenvalues, eigenvectors

@jax.jit
def get_model(
        t: jax.Array, 
        d: jax.Array,
        signals
    ):
    '''
    Bretthorst Eq. 3.5 and Eq. 3.13
    '''

    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    coefficients = jnp.linalg.lstsq(G.T, d, rcond=None)[0]

    return G.T @ coefficients

def get_noise_variance(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ):
    '''
    Bretthorst Eq. 4.7
    '''

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
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    return (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

def get_snr(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ):
    '''
    Bretthorst Eq. 4.8
    '''
    
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
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)

    return ((m / N) * (1 + ((1 / m) * sum_sq_proj / noise_variance))) ** (1 / 2)

def get_mean_sq_proj(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ) -> jax.Array:
    '''
    Bretthorst Eq. 3.15
    '''

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
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    return jnp.mean(h ** 2)

def get_glob_ll(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ) -> jax.Array:
    '''
    Bretthorst Eq. 5.9
    '''

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

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T
    g = 0.5 * (g + g.T)

    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d

    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)
    mean_sq_param = (1 / r) * jnp.sum(omegas ** 2 + ks ** 2)

    coefficients = jnp.linalg.lstsq(G.T, d, rcond=None)[0]
    model = G.T @ coefficients
    residual = d - model

    eps = jnp.finfo(d.dtype).eps

    amplitude_max = jnp.maximum(jnp.max(jnp.abs(d)), eps)
    amplitude_min = jnp.maximum(jnp.quantile(jnp.abs(d), 0.05), eps)

    variance_data = jnp.maximum(jnp.var(d), eps)
    variance_residual = jnp.maximum(jnp.var(residual), eps)

    log_R_delta = 2.0 * jnp.log(amplitude_max / amplitude_min)
    log_R_gamma = jnp.log(1e3)
    log_R_sigma = jnp.log(jnp.maximum(variance_data / variance_residual, 1.0))

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = utils.get_h_eigendecomposition(b)

    log_jacobian_factor = -0.5 * jnp.sum(jnp.log(eigenvalues))

    delta_term = (
        jsp.gammaln(m / 2)
        - jnp.log(2) - log_R_delta
        + (-m / 2) * jnp.log(m * mean_sq_proj / 2.0)
    )

    gamma_term = (
        jsp.gammaln(r / 2)
        - jnp.log(2) - log_R_gamma
        + (-r / 2) * jnp.log((r * mean_sq_param) / 2)
    )

    sigma_term = (
        jsp.gammaln((N - m - r) / 2)
        - jnp.log(2) - log_R_sigma
        + ((m + r - N) / 2) * jnp.log(
            ((N * mean_sq_data) - (m * mean_sq_proj)) / 2
        )
    )

    return delta_term + sigma_term + gamma_term + log_jacobian_factor

def get_phasor_parameters(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ):
    '''
    Definition of model functions
    '''

    fs, ks = utils.unpack_signals(signals)

    omegas = 2.0 * jnp.pi * fs
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))
    g = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

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


def get_cov_mat(t, d, signals):
    '''
    Return covariance matrix for parameters.
    '''
    fs, ks = utils.unpack_signals(signals)
    r = fs.shape[0]
    m = 2 * r
    N = d.shape[0]

    theta, unravel = ravel_pytree(signals)
    
    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = utils.get_h_eigendecomposition(b)

    information_inverse = (
        eigenvectors
        @ jnp.diag(1.0 / eigenvalues)
        @ eigenvectors.T
    )

    # Use the same projection calculations as get_noise_variance.
    noise_variance = get_noise_variance(t, d, signals)

    return noise_variance * information_inverse

def get_uncertainties(
        t: jax.Array, 
        d: jax.Array, 
        signals
    ) -> jax.Array:
    '''
    Bretthorst Eq. 4.13
    '''

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
    eigenvalues, eigenvectors = utils.get_g_eigendecomposition(g)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    theta, unravel = ravel_pytree(signals)

    def objective(theta):
        return get_mean_sq_proj(t, d, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    eigenvalues, eigenvectors = utils.get_h_eigendecomposition(b)

    sum_sq_data = jnp.sum(d**2)
    sum_sq_proj = jnp.sum(h**2)

    noise_variance = (
        sum_sq_data - sum_sq_proj
    ) / (N - m - 2)

    variance_per_parameter = noise_variance * jnp.sum(
        eigenvectors**2 / eigenvalues[None, :],
        axis=1,
    )

    signals_uncertainties_flat = jnp.sqrt(
        jnp.maximum(variance_per_parameter, 0.0)
    )

    signals_uncertainties = unravel(signals_uncertainties_flat)

    return signals_uncertainties

def reconcile(
        t,
        d,
        signal_space,
        signals,
        signals_bounds,
        k2_threshold=10,
        nuts_args=None,
    ):
    '''
    Procedure for removing degenerate signals.
    1) Identify smallest eigenvalue (most dependent / smallest amplitude signal).
    2) Evaluate signals for largest contribution to dependent direction.
    3) Remove most impactful signal to probability.
    '''

    signals = jnp.asarray(signals)
    signals_bounds = jnp.asarray(signals_bounds)

    while len(signals) > 1:
        gram, _, _ = get_gram(t, d, signals)

        norms = jnp.sqrt(jnp.diag(gram))
        normalized_gram = gram / jnp.outer(norms, norms)

        eigenvalues, eigenvectors = jnp.linalg.eigh(normalized_gram)

        max_eigenvalue = jnp.max(eigenvalues)
        min_eigenvalue = jnp.maximum(jnp.min(eigenvalues), jnp.finfo(eigenvalues.dtype).eps * max_eigenvalue)

        # Gram eigenvalues are squared singular values.
        k2 = jnp.sqrt(max_eigenvalue / min_eigenvalue)
        print(round(k2, 3))
        if k2 <= k2_threshold:
            break

        eigenvector = eigenvectors[:, jnp.argmin(eigenvalues)]
        m = len(signals)

        cosine_components = eigenvector[:m]
        sine_components = eigenvector[m:]

        signal_strength = jnp.sqrt(
            cosine_components ** 2 + sine_components ** 2
        )

        candidate_indices = jnp.argsort(signal_strength)[-4:]

        candidate_scores = []

        for candidate_index in candidate_indices:
            candidate_index = int(candidate_index)
            candidate_signals = jnp.delete(signals, candidate_index, axis=0)
            score = get_log_prob(t, d, candidate_signals)

            candidate_scores.append((score, candidate_index))

        _, signal_index = max(
            candidate_scores,
            key=lambda result: float(result[0])
        )

        f, k = signals[signal_index]
        print(
            f"Removed signal with frequency {round(f, 8)} Hz and decay rate {round(k, 8)}. "
            f"Condition number: {round(k2, 5)}."
        )

        signals = jnp.delete(signals, signal_index, axis=0)
        signals_bounds = jnp.delete(signals_bounds, signal_index, axis=0)

        if nuts_args is not None:
            signals = nuts(
                t, 
                d, 
                signal_space, 
                signals, 
                signals_bounds,
                nuts_args.nuts_kwargs, 
                nuts_args.mcmc_kwargs,
                nuts_args.run_kwargs, 
                nuts_args.seed
            )

    return signals, signals_bounds

def grid_search(
        t, 
        d, 
        signal_space,
        f_points, 
        k_points,  
        return_probability_surface=False, 
        batch_size=256
    ):
    '''
    Procedure for identifying candidate signal with a grid search.
    1) Create grid across chosen parameter set of frequencies and decay rates.
    2) Evaluate probability across the grid.
    3) Return signal with highest probability.
    '''

    f_min, f_max, k_min, k_max = signal_space.f_min, signal_space.f_max, signal_space.k_min, signal_space.k_max

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
    f_min: jax.Array,
    f_max: jax.Array,
    k_min: jax.Array,
    k_max: jax.Array,
) -> None:
    '''
    Identify sampling space and method for NUTS.
    '''
    
    f_min = jnp.atleast_1d(jnp.asarray(f_min))
    f_max = jnp.broadcast_to(
        jnp.asarray(f_max, dtype=f_min.dtype),
        f_min.shape,
    )

    k_min = jnp.atleast_1d(jnp.asarray(k_min))
    k_max = jnp.broadcast_to(
        jnp.asarray(k_max, dtype=k_min.dtype),
        k_min.shape,
    )

    log_k_min = jnp.log(k_min)
    log_k_max = jnp.log(k_max)

    fs = numpyro.sample(
        "fs",
        dist.Uniform(f_min, f_max).to_event(1),
    )

    log_ks = numpyro.sample(
        "log_ks",
        dist.Uniform(log_k_min, log_k_max).to_event(1),
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
    signals_bounds,
    nuts_kwargs,
    mcmc_kwargs,
    run_kwargs,
    rng_key_value,
):
    '''
    Run NUTS for chosen signals.
    '''
    
    f_init, k_init = utils.unpack_signals(signals)

    f_init = jnp.atleast_1d(jnp.asarray(f_init))
    k_init = jnp.atleast_1d(jnp.asarray(k_init))

    bounds = jnp.asarray(signals_bounds)

    if bounds.ndim != 2 or bounds.shape != (f_init.size, 4):
        raise ValueError(
            "signals_bounds must have shape (n_signals, 4), with entries "
            "(f_min, f_max, k_min, k_max)"
        )

    fs_min = bounds[:, 0].astype(f_init.dtype)
    fs_max = bounds[:, 1].astype(f_init.dtype)
    ks_min = bounds[:, 2].astype(k_init.dtype)
    ks_max = bounds[:, 3].astype(k_init.dtype)

    if bool(jnp.any(fs_min >= fs_max)):
        raise ValueError(
            f"Empty frequency intervals: low={fs_min}, high={fs_max}"
        )

    if bool(jnp.any(ks_min <= 0)):
        raise ValueError(
            f"Decay-rate lower bounds must be positive: {ks_min}"
        )

    if bool(jnp.any(ks_min >= ks_max)):
        raise ValueError(
            f"Empty decay-rate intervals: low={ks_min}, high={ks_max}"
        )

    # The persistent representation is physical k. Convert to log(k)
    # only for the NUTS sampling distribution.
    log_k_init = jnp.log(k_init)
    log_ks_min = jnp.log(ks_min)
    log_ks_max = jnp.log(ks_max)

    f_width = fs_max - fs_min
    log_k_width = log_ks_max - log_ks_min

    f_eps = 1e-6 * jnp.maximum(f_width, 1.0)
    log_k_eps = 1e-6 * jnp.maximum(log_k_width, 1.0)

    f_init_safe = jnp.clip(
        f_init,
        fs_min + f_eps,
        fs_max - f_eps,
    )

    log_k_init_safe = jnp.clip(
        log_k_init,
        log_ks_min + log_k_eps,
        log_ks_max - log_k_eps,
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
        rng_key_value = random.randint(1, 10_000)
    else:
        rng_key_value = rng_key_value + len(signals)

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
        f_min=fs_min,
        f_max=fs_max,
        k_min=ks_min,
        k_max=ks_max,
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
    '''
    Experimental minimization function to improve grid search / NUTS results.
    '''

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
