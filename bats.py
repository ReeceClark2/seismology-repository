# Standard library
import csv
import multiprocessing as mp
import os
import random
import shutil
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import traceback
from time import perf_counter

# Configure Matplotlib before importing pyplot.
os.environ["MPLBACKEND"] = "Agg"

# Third-party
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import matplotlib
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
from matplotlib.patches import Ellipse
from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import brentq, minimize
from scipy.signal import butter, sosfiltfilt, sosfreqz

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

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

# Linear algebra helpers -----------------------------------------------------------------

# Relative ridge added to the unit-diagonal (equilibrated) Gram matrix before Cholesky.
# Because the residual is recomputed explicitly, this biases the RSS only at second order.
GRAM_JITTER = 1e-12

# Filtering: trim each end until filter ringing is below DEFAULT_RINGING_DB (amplitude, relative
# to the transient's peak), never less than MIN_TRIM_SECONDS and never more than the cap.
# The cap protects the early, highest-SNR part of the record: trimming T_trim scales a mode's
# amplitude at the new t = 0 by exp(-pi f T_trim / Q), which costs low-Q modes most.
MIN_TRIM_SECONDS = 1 * 3600

# Bretthorst global likelihood: bound ratios R (upper / lower) of the Jeffreys priors on the
# amplitude scale delta and the noise sigma. They enter only as log(log R); delta matters only
# when comparing s = 0 with s >= 1.
BRETTHORST_RATIO_DELTA = 5e1

BRETTHORST_RATIO_SIGMA = 1e2
DEFAULT_MAX_TRIM_SECONDS = 10 * 3600
DEFAULT_RINGING_DB = -60.0

# Subband filtering: Butterworth order, and the fraction of a signal's amplitude the zero-phase
# filter must keep at the core-band edges (the filter band is widened until it does).
DEFAULT_FILTER_ORDER = 8
DEFAULT_CORE_GAIN = 0.99

# Largest allowed 2-norm condition number of the normalized basis, sqrt(lambda_max / lambda_min) of the
# unit-diagonal Gram matrix. Above it the newest signal is (numerically) a combination of the others:
# 1,000 is an eigenvalue ratio of 1e6, an overlap of about 1 - 1e-6 for two signals, far beyond any
# pair the record can resolve. Such a model is invalid and ends the greedy search.
DEFAULT_MAX_CONDITION = 1e3

# Fit band: each side extends from the filter edge until the zero-phase amplitude gain |H|^2 falls
# to this value (1e-3 = -60 dB), so a mode beyond the fit band leaks in at no more than that
# fraction of its amplitude. Set per side: the upper skirt falls more slowly in linear frequency.
DEFAULT_FIT_EDGE_GAIN = 1e-3

# Model error: what the model class cannot represent (leakage, filter artifacts, departures from
# pure damped sinusoids), in dB relative to the subband's mean power. The BIC never assumes a
# noise variance below 10^(model_error_db / 10) mean(d^2), so the fit is not credited with
# structure below the pipeline's own error. Inactive whenever the real noise is above it.
DEFAULT_MODEL_ERROR_DB = -50.0
NOVELTY_JITTER = 1e-10

def get_basis(t, fs, Qs):
    """Complex basis E_j(t) = exp(z_j t), z_j = -pi f_j / Q_j + 2 pi i f_j.

    The real model basis is G = [Re(E); Im(E)], i.e. rows [cos * decay; sin * decay].
    """
    z = -jnp.pi * fs / Qs + 2j * jnp.pi * fs
    E = jnp.exp(z[:, None] * t[None, :])
    return z, E

def get_cross_gram(t, z1, z2):
    """Analytic inner products between the real bases of z1 and z2 (uniform sampling).

    Returns the (2 s1, 2 s2) block [[cc, cs], [sc, ss]] with rows (cos_1, sin_1) and
    columns (cos_2, sin_2), computed from geometric sums in O(s1 s2).
    """
    n = t.shape[0]
    dt = t[1] - t[0]

    def geometric_sum(u):
        q = u * dt
        q = jnp.real(q) + 1j * (jnp.imag(q) - 2 * jnp.pi * jnp.round(jnp.imag(q) / (2 * jnp.pi)))
        denominator = jnp.where(q == 0, 1, jnp.expm1(q))
        series = jnp.where(q == 0, n, jnp.expm1(n * q) / denominator)
        return jnp.exp(u * t[0]) * series

    A = geometric_sum(z1[:, None] + z2[None, :])
    C = geometric_sum(z1[:, None] + jnp.conj(z2[None, :]))

    cc = 0.5 * jnp.real(A + C)
    cs = 0.5 * jnp.imag(A - C)
    sc = 0.5 * jnp.imag(A + C)
    ss = 0.5 * jnp.real(C - A)
    return jnp.block([[cc, cs], [sc, ss]])

def get_gram(t, z):
    g = get_cross_gram(t, z, z)
    return 0.5 * (g + g.T)

def get_basis_condition(t, signals):
    """2-norm condition number of the column-normalized basis [cos * decay, sin * decay]:
    sqrt(lambda_max / lambda_min) of the unit-diagonal Gram matrix (Gram eigenvalues are squared
    singular values). 1 for an orthogonal basis; it grows as signals become linearly dependent.
    Uses the analytic Gram matrix, so it costs O(s^2) plus a (2s x 2s) eigendecomposition.
    """
    fs, Qs = unpack_signals(signals, sort=False)
    if fs.shape[0] == 0:
        return 1.0
    z = -jnp.pi * fs / Qs + 2j * jnp.pi * fs
    g = get_gram(t, z)
    norms = jnp.sqrt(jnp.maximum(jnp.diag(g), jnp.finfo(g.dtype).tiny))
    g_normalized = g / jnp.outer(norms, norms)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (g_normalized + g_normalized.T))
    max_eigenvalue = jnp.maximum(jnp.max(eigenvalues), jnp.finfo(eigenvalues.dtype).tiny)
    min_eigenvalue = jnp.maximum(jnp.min(eigenvalues), jnp.finfo(eigenvalues.dtype).eps * max_eigenvalue)
    return float(jnp.sqrt(max_eigenvalue / min_eigenvalue))

def get_equilibrated_cholesky(g, jitter=GRAM_JITTER):
    """Lower Cholesky factor of D^-1 g D^-1 + jitter I, with D = sqrt(diag(g))."""
    D = jnp.sqrt(jnp.maximum(jnp.diag(g), jnp.finfo(g.dtype).tiny))
    g_scaled = g / jnp.outer(D, D) + jitter * jnp.eye(g.shape[0], dtype=g.dtype)
    return jnp.linalg.cholesky(g_scaled), D

def solve_amplitudes(g, b):
    """Solve g beta = b with Jacobi equilibration and a small relative ridge."""
    D = jnp.sqrt(jnp.maximum(jnp.diag(g), jnp.finfo(g.dtype).tiny))
    g_scaled = g / jnp.outer(D, D) + GRAM_JITTER * jnp.eye(g.shape[0], dtype=g.dtype)
    return cho_solve(cho_factor(g_scaled), b / D) / D

def get_projection_pieces(t, d, fs, Qs):
    """Basis, amplitudes and explicit residual of the variable-projection fit.

    Returns (z, E, c, r): E_j(t) = exp(z_j t); c_j = beta_cos,j - i beta_sin,j, so the fitted model is
    Re(sum_j c_j E_j) = G^T beta with G = [Re(E); Im(E)]; r = d - model.
    """
    z, E = get_basis(t, fs, Qs)
    Ed = E @ d                                            # (G d) as cos.d + i sin.d, one pass
    beta = solve_amplitudes(get_gram(t, z), jnp.concatenate((jnp.real(Ed), jnp.imag(Ed))))
    k = fs.shape[0]
    c = beta[:k] - 1j * beta[k:]
    r = d - jnp.real(E.T @ c)                             # G^T beta, one pass
    return z, E, c, r

@jax.custom_jvp
def get_stable_rss(t, d, fs, Qs):
    """Residual sum of squares computed from an explicit residual vector.

    RSS = ||d - G^T beta||^2 is stationary in beta, so conditioning errors in beta only
    enter at second order, unlike sum(d^2) - b^T g^-1 b which loses ~kappa * eps * sum(d^2).
    Cost O(s n + s^3).

    Derivatives use the variable-projection identity (see get_stable_rss_jvp) instead of tracing
    autodiff through the exponentials, matrix products and Cholesky solve: about 2-4x faster
    value-plus-gradient, and second derivatives (Hessian-vector products) still work.
    """
    return jnp.sum(get_projection_pieces(t, d, fs, Qs)[3] ** 2)

@get_stable_rss.defjvp
def get_stable_rss_jvp(primals, tangents):
    """Variable projection: beta minimizes the RSS, so dRSS/dp = -2 sum_t r(t) dm/dp at fixed beta
    (the beta dependence drops out at first order). With m = Re(sum_j c_j E_j) and dE_j/dz_j = t E_j:

        dRSS/df_j = -2 Re[c_j (dz_j/df) (E (t r))_j],   dRSS/dQ_j = -2 Re[c_j (dz_j/dQ) (E (t r))_j],

    with dz/df = -pi / Q + 2 pi i and dz/dQ = pi f / Q^2: one extra matrix product. Exact up to the
    1e-12 ridge in the amplitude solve. Not differentiable in t or d (their tangents are ignored).
    """
    t, d, fs, Qs = primals
    _, _, dfs, dQs = tangents
    z, E, c, r = get_projection_pieces(t, d, fs, Qs)
    w = c * (E @ (t * r))
    grad_fs = -2.0 * jnp.real(w * (-jnp.pi / Qs + 2j * jnp.pi))
    grad_Qs = -2.0 * jnp.real(w * (jnp.pi * fs / Qs ** 2))
    return jnp.sum(r ** 2), jnp.sum(grad_fs * dfs) + jnp.sum(grad_Qs * dQs)

def get_posterior_variance(t, d, B, signals):
    """v = RSS / (N - m - 2), the noise scale of the Bretthorst covariance (N = 2BT).

    Proportional to the in-band noise level and independent of bandwidth, so values from
    different subbands and from the full band are comparable.
    """
    fs, Qs = unpack_signals(signals, sort=False)
    s = fs.shape[0]
    m = 2 * s
    N = int(2 * t[-1] * B)
    rss = jnp.sum(d ** 2) if s == 0 else get_stable_rss(t, d, fs, Qs)
    return float(rss / max(N - m - 2.0, 1.0))

def get_subband_index(fs, band_edges):
    """Core subband that owns each frequency: [edge_k, edge_k+1), the last core closed above.

    Frequencies outside [edge_0, edge_-1] are assigned to the nearest end subband.
    """
    edges = np.asarray(band_edges)
    return np.clip(np.searchsorted(edges, np.asarray(fs), side="right") - 1, 0, len(edges) - 2)

def get_subband_scaled_covariance(signals, covariance, band_edges, subband_variances, full_variance):
    """Rescale a full-band Bretthorst covariance to each signal's subband noise level.

    Sigma_ij(local) = Sigma_ij(joint) * sqrt(v_k(i) v_k(j)) / v_full. Correlations are kept;
    variances follow the locally-white approximation. Each signal's subband k(i) is the core
    band that owns its frequency. Rows follow get_statistics' f-sorted, interleaved
    [f1, Q1, f2, Q2, ...] order.
    """
    signals = np.asarray(signals).reshape(-1, 2)
    subband_index = get_subband_index(np.sort(signals[:, 0]), band_edges)

    signal_scale = np.sqrt(np.asarray(subband_variances)[subband_index] / full_variance)
    parameter_scale = np.repeat(signal_scale, 2)
    scaled_covariance = np.asarray(covariance) * np.outer(parameter_scale, parameter_scale)

    return jnp.asarray(scaled_covariance), signal_scale, subband_index

def save_joint_uncertainties(path, fs, Qs, fs_unc_full, Qs_unc_full, signal_scale, subband_index):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)

    with (diagnostic_path / "bretthorst_uncertainties.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Frequency", "Quality Factor", "Subband",
            "Frequency Uncertainty (subband sigma)", "Quality Factor Uncertainty (subband sigma)",
            "Frequency Uncertainty (full-band sigma)", "Quality Factor Uncertainty (full-band sigma)",
            "Noise Scale Ratio sqrt(v_subband / v_full)",
        ])
        writer.writerows(zip(
            np.asarray(fs), np.asarray(Qs), subband_index + 1,
            np.asarray(fs_unc_full) * signal_scale, np.asarray(Qs_unc_full) * signal_scale,
            np.asarray(fs_unc_full), np.asarray(Qs_unc_full),
            signal_scale,
        ))

def interleaved_indices(signal_indices):
    """Parameter indices [2i, 2i + 1, ...] of the given signals in an interleaved (f, log Q) layout."""
    signal_indices = np.asarray(signal_indices, dtype=int)
    return np.stack((2 * signal_indices, 2 * signal_indices + 1), axis=-1).ravel()

def prune_signals(t, d, B, signals, bounds, f_min, f_max, Q_min, Q_max, max_condition=DEFAULT_MAX_CONDITION, model_error_db=DEFAULT_MODEL_ERROR_DB):
    """Backward elimination: drop the signal whose removal most improves the model-selection
    score (the BIC from get_statistics), repeat.

    Remaining signals are not re-optimized after a removal, so each removal score is a
    lower bound on what a refit would give (the pruning is conservative).
    Cost per pass O(s * n s^2): one get_statistics (QR + Fisher) per candidate removal.
    """
    def score(candidate):
        return float(get_statistics(t, d, B, jnp.asarray(candidate), f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)[-1])

    signals = np.array(signals, dtype=float).reshape(-1, 2)
    bounds = np.array(bounds, dtype=float).reshape(-1, 4)
    current = score(signals)
    removed = []

    while signals.shape[0] > 1:
        scores = np.array([score(np.delete(signals, j, axis=0)) for j in range(signals.shape[0])])
        j = int(np.argmax(scores))

        if not scores[j] > current:
            break

        removed.append(tuple(signals[j]))
        signals = np.delete(signals, j, axis=0)
        bounds = np.delete(bounds, j, axis=0)
        current = scores[j]

    return jnp.asarray(signals), jnp.asarray(bounds), removed

# Signal Update --------------------------------------------------------------------------

def get_log_prob(t, d, B, signals):
    # Permutation-invariant, so no sorting is needed inside the objective.
    fs, Qs = unpack_signals(signals, sort=False)
    m = 2 * fs.shape[0]
    N = int(2 * t[-1] * B)

    sum_sq_data = jnp.sum(d ** 2)
    rss = get_stable_rss(t, d, fs, Qs)
    rss = jnp.maximum(rss, jnp.finfo(d.dtype).eps * sum_sq_data)

    # Bretthorst: [1 - m h^2 / (N d^2)]^((m - N) / 2) = (RSS / sum d^2)^((m - N) / 2)
    return 0.5 * (m - N) * jnp.log(rss / sum_sq_data)

def get_novelty_function(t, existing):
    """Return f(signal) = 1 - ||P_existing e||^2 / ||e||^2 for a candidate's (cos, sin) pair.

    Uses analytic cross-Gram entries, so each candidate costs O(s^2) after one
    O(s^3) factorization. Returns None if there is nothing to compare against, or if the
    existing Gram matrix cannot be factorized (check disabled for that iteration).
    """
    existing = jnp.asarray(existing).reshape(-1, 2)

    if existing.shape[0] == 0:
        return None

    e_fs, e_Qs = existing[:, 0], existing[:, 1]
    z_e = -jnp.pi * e_fs / e_Qs + 2j * jnp.pi * e_fs
    L_e, D_e = get_equilibrated_cholesky(get_gram(t, z_e), NOVELTY_JITTER)

    if not bool(jnp.all(jnp.isfinite(L_e))):
        print("Warning: existing Gram matrix not factorizable; skipping novelty check.", flush=True)
        return None

    def novelty(signal):
        fs_c, Qs_c = unpack_signals(signal, sort=False)
        z_c = -jnp.pi * fs_c / Qs_c + 2j * jnp.pi * fs_c
        candidate_norm_sq = jnp.trace(get_gram(t, z_c))
        X = get_cross_gram(t, z_e, z_c) / D_e[:, None]
        Y = solve_triangular(L_e, X, lower=True)
        return 1.0 - jnp.sum(Y ** 2) / candidate_norm_sq

    return novelty

def grid_search(t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, batch_size=512, existing=None, min_novelty=0.0, B=None):
    f_space = jnp.linspace(f_min, f_max, f_points)
    # B is the filter bandwidth (sets N = 2BT), not the searched frequency range.
    B = f_max - f_min if B is None else B

    Q_margin = 1e-6 * (Q_max - Q_min)
    Q_space = jnp.geomspace(Q_min + Q_margin, Q_max - Q_margin, Q_points)

    f_grid, Q_grid = jnp.meshgrid(f_space, Q_space, indexing="ij")
    signal_grid = jnp.stack([f_grid.ravel(), Q_grid.ravel()], axis=-1)

    evaluate_batch = jax.jit(
        jax.vmap(lambda signal: get_log_prob(t, d, B, signal))
    )

    novelty_function = None
    if existing is not None and min_novelty > 0:
        novelty_function = get_novelty_function(t, existing)

    evaluate_novelty = jax.jit(jax.vmap(novelty_function)) if novelty_function is not None else None

    results = []
    novelties = []

    for start in range(0, signal_grid.shape[0], batch_size):
        batch = signal_grid[start:start + batch_size]
        results.append(evaluate_batch(batch))
        if evaluate_novelty is not None:
            novelties.append(evaluate_novelty(batch))

    log_probs = jnp.concatenate(results)

    # Candidates mostly inside the span of existing signals are not resolvable; mask them.
    if evaluate_novelty is not None:
        novelty = jnp.concatenate(novelties)
        log_probs = jnp.where(novelty >= min_novelty, log_probs, -jnp.inf)

    log_prob_space = log_probs.reshape(f_space.size, Q_space.size)

    probability_surface = (f_space, Q_space, log_prob_space)
    
    return probability_surface

def objective(t, d, B, f_min, f_max, Q_min, Q_max):
    fs = numpyro.sample("fs", numpyro.distributions.Uniform(f_min, f_max).to_event(1))
    log_Qs = numpyro.sample("log_Qs", numpyro.distributions.Uniform(jnp.log(Q_min), jnp.log(Q_max)).to_event(1))

    Qs = numpyro.deterministic("Qs", jnp.exp(log_Qs))
    signals = jnp.stack((fs, Qs), axis=-1)

    numpyro.factor("surface", get_log_prob(t, d, B, signals))
 
MAP_BOUND_TOLERANCE = 1e-3
MAX_TREE_DEPTH = 6

def map_refine(t, d, B, signals, bounds, method="trust-krylov", max_iterations=1000, gtol=1e-6):
    """Refine a starting point to the posterior mode (MAP) within the NUTS prior box.

    The NUTS prior is uniform in (f, log Q) on each signal's box, so the MAP in those
    coordinates is argmax get_log_prob. The box constraint is enforced by reparameterizing:
    the optimizer works on unconstrained u, with x = lower + width * sigmoid(u), so every
    step stays inside the box. The objective omits the change-of-variables Jacobian, so its
    optimum maps to the MAP in (f, log Q), not to the peak of the density in u.

    Second-order trust-region Newton with exact curvature:
      - "trust-krylov" (default): exact Hessian-vector products (forward-over-reverse),
        Lanczos/GLTR subproblem. Handles indefinite curvature away from the mode and needs
        memory of only a few gradients, independent of the number of parameters.
      - "trust-exact": dense exact Hessian assembled column by column from HVPs
        (sequential lax.map, so memory stays bounded). O(2s) HVPs per iteration; better
        for small s.

    The result is accepted only if it does not lower the log probability.
    """
    fs0, Qs0 = unpack_signals(signals, sort=False)
    bounds = jnp.asarray(bounds).reshape(-1, 4)
    s = fs0.shape[0]

    lower = jnp.concatenate((bounds[:, 0], jnp.log(bounds[:, 2])))
    upper = jnp.concatenate((bounds[:, 1], jnp.log(bounds[:, 3])))
    width = upper - lower
    x0 = jnp.concatenate((fs0, jnp.log(Qs0)))
    p0 = jnp.clip((x0 - lower) / width, MAP_BOUND_TOLERANCE, 1.0 - MAP_BOUND_TOLERANCE)
    u0 = jnp.log(p0) - jnp.log1p(-p0)

    def to_signals(u):
        x = lower + width * jax.nn.sigmoid(u)
        return jnp.stack((x[:s], jnp.exp(x[s:])), axis=-1)

    def negative_log_prob(u):
        return -get_log_prob(t, d, B, to_signals(u))

    gradient = jax.grad(negative_log_prob)
    value_and_grad = jax.jit(jax.value_and_grad(negative_log_prob))

    def hvp_raw(u, v):
        return jax.jvp(gradient, (u,), (v,))[1]

    hvp = jax.jit(hvp_raw)

    def fun(u):
        value, grad = value_and_grad(jnp.asarray(u))
        value = float(value)
        if not np.isfinite(value):
            # Non-finite trial points (e.g. Cholesky failure) are rejected by the trust region.
            return np.inf, np.zeros_like(u)
        return value, np.asarray(grad, dtype=np.float64)

    start_value = fun(np.asarray(u0, dtype=np.float64))[0]
    kwargs = {}

    if method == "trust-exact":
        eye = jnp.eye(2 * s, dtype=u0.dtype)
        dense_hessian = jax.jit(lambda u: jax.lax.map(lambda v: hvp_raw(u, v), eye))

        def hess(u):
            H = np.asarray(dense_hessian(jnp.asarray(u)), dtype=np.float64)
            return 0.5 * (H + H.T)

        kwargs["hess"] = hess
    else:
        kwargs["hessp"] = lambda u, v: np.asarray(hvp(jnp.asarray(u), jnp.asarray(v)), dtype=np.float64)

    try:
        result = minimize(fun, np.asarray(u0, dtype=np.float64), jac=True, method=method, options={"maxiter": max_iterations, "gtol": gtol}, **kwargs)
    except Exception as error:
        print(f"MAP refinement failed ({error}); keeping the starting point.", flush=True)
        return jnp.stack((fs0, Qs0), axis=-1), {"accepted": False, "improvement": 0.0, "iterations": 0, "message": str(error), "at_bound": 0, "free_gradient": float("nan"), "bound_gradient": float("nan")}

    accepted = bool(np.isfinite(result.fun) and result.fun <= start_value)
    u_final = jnp.asarray(result.x) if accepted else u0
    p_final = jax.nn.sigmoid(u_final)
    near_bound = (p_final < MAP_BOUND_TOLERANCE) | (p_final > 1.0 - MAP_BOUND_TOLERANCE)
    signal_at_bound = near_bound[:s] | near_bound[s:]
    at_bound = int(jnp.sum(signal_at_bound))

    # Convergence check (one extra gradient): d log p / d(x / box width) in (f, log Q),
    # so f and log Q are comparable, in nats per box width. Near zero = stationary.
    def log_prob_x(x):
        return get_log_prob(t, d, B, jnp.stack((x[:s], jnp.exp(x[s:])), axis=-1))

    x_final = lower + width * p_final
    scaled_gradient = jnp.abs(jax.grad(log_prob_x)(x_final) * width)
    signal_gradient = jnp.maximum(scaled_gradient[:s], scaled_gradient[s:])
    free_gradient = float(jnp.max(jnp.where(signal_at_bound, 0.0, signal_gradient))) if at_bound < s else float("nan")
    bound_gradient = float(jnp.max(jnp.where(signal_at_bound, signal_gradient, 0.0))) if at_bound > 0 else float("nan")

    info = {
        "accepted": accepted,
        "improvement": float(start_value - result.fun) if accepted else 0.0,
        "iterations": int(result.nit),
        "message": str(result.message),
        "at_bound": at_bound,
        "free_gradient": free_gradient,
        "bound_gradient": bound_gradient,
    }

    return to_signals(u_final), info

def get_nuts_diagnostics(fs, log_Qs, extra_fields, max_tree_depth=MAX_TREE_DEPTH):
    """Sampler diagnostics on f-sorted draws, arrays shaped (chains, draws, s).

    Sorting each draw by f is a relabelling that removes most label switching, so per-index
    ESS and split R-hat are meaningful except where neighbouring signals genuinely swap.
    """
    fs = np.asarray(fs)
    log_Qs = np.asarray(log_Qs)
    Qs = np.exp(log_Qs)
    n_chains, n_draws, n_signals = fs.shape

    diverging = np.asarray(extra_fields["diverging"])
    num_steps = np.asarray(extra_fields["num_steps"])
    accept_prob = np.asarray(extra_fields["accept_prob"])

    def ess(x):
        return np.asarray(effective_sample_size(x)) if n_draws >= 2 else np.full(n_signals, np.nan)

    def rhat(x):
        return np.asarray(split_gelman_rubin(x)) if n_draws >= 4 else np.full(n_signals, np.nan)

    ddof = 1 if n_chains * n_draws > 1 else 0

    return {
        "chains": n_chains,
        "draws_per_chain": n_draws,
        "divergent": int(np.sum(diverging)),
        "divergent_fraction": float(np.mean(diverging)),
        "max_tree_depth_fraction": float(np.mean(num_steps >= 2 ** max_tree_depth - 1)),
        "mean_accept_prob": float(np.mean(accept_prob)),
        "f_mean": fs.mean(axis=(0, 1)),
        "f_sd": fs.std(axis=(0, 1), ddof=ddof),
        "Q_mean": Qs.mean(axis=(0, 1)),
        "Q_sd": Qs.std(axis=(0, 1), ddof=ddof),
        "ess_f": ess(fs),
        "ess_log_Q": ess(log_Qs),
        "rhat_f": rhat(fs),
        "rhat_log_Q": rhat(log_Qs),
    }

def nuts(t, d, B, signals, bounds, warmup, sample, chains, rng_key_value=None, refine=True, return_diagnostics=False, tag=""):
    f_init, Q_init = unpack_signals(signals, sort=False)
    bounds = jnp.asarray(bounds).reshape(-1, 4)
    fs_min = bounds[:, 0].astype(f_init.dtype)
    fs_max = bounds[:, 1].astype(f_init.dtype)
    Qs_min = bounds[:, 2].astype(Q_init.dtype)
    Qs_max = bounds[:, 3].astype(Q_init.dtype)
    init_strategy = init_to_value(values={"fs": f_init, "log_Qs": jnp.log(Q_init)})

    if rng_key_value is None:
        rng_key_value = random.randint(1, 10_000)

    warmup_key, _ = jax.random.split(jax.random.PRNGKey(int(rng_key_value)))
    kernel = NUTS(objective, init_strategy=init_strategy, max_tree_depth=MAX_TREE_DEPTH)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=sample, num_chains=chains, chain_method="parallel", progress_bar=False)
    mcmc.warmup(warmup_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, Q_min=Qs_min, Q_max=Qs_max)
    mcmc.run(mcmc.post_warmup_state.rng_key, t=t, d=d, B=B, f_min=fs_min, f_max=fs_max, Q_min=Qs_min, Q_max=Qs_max, extra_fields=("potential_energy", "diverging", "accept_prob", "num_steps"))

    samples = mcmc.get_samples(group_by_chain=True)
    extra_fields = mcmc.get_extra_fields(group_by_chain=True)
    fs = samples["fs"]
    log_Qs = samples["log_Qs"]

    order = jnp.argsort(fs, axis=-1)
    fs = jnp.take_along_axis(fs, order, axis=-1)
    log_Qs = jnp.take_along_axis(log_Qs, order, axis=-1)
    draws = jnp.stack((fs, log_Qs), axis=-1).reshape(-1, f_init.shape[0], 2)
    orders = order.reshape(-1, f_init.shape[0])

    valid = jnp.isfinite(extra_fields["potential_energy"]) & ~extra_fields["diverging"]
    valid = valid.reshape(-1)

    if not bool(jnp.any(valid)):
        raise ValueError("NUTS produced no finite, non-divergent post-warmup draws.")

    draws = draws[valid]
    orders = orders[valid]

    # Start from the draw with the highest log probability (closest to the mode). The NUTS
    # potential energy includes the unconstraining Jacobian, so get_log_prob is evaluated
    # directly; lax.map keeps memory at one evaluation at a time.
    candidate_signals = jnp.stack((draws[..., 0], jnp.exp(draws[..., 1])), axis=-1)
    draw_log_probs = jax.lax.map(lambda candidate: get_log_prob(t, d, B, candidate), candidate_signals)
    draw_log_probs = jnp.where(jnp.isfinite(draw_log_probs), draw_log_probs, -jnp.inf)
    selected_index = jnp.argmax(draw_log_probs)
    selected = draws[selected_index]
    selected_order = orders[selected_index]

    returned_signals = jnp.stack((selected[:, 0], jnp.exp(selected[:, 1])), axis=-1)
    returned_bounds = bounds[selected_order]

    if refine:
        returned_signals, map_info = map_refine(t, d, B, returned_signals, returned_bounds)
        prefix = f"{tag} " if tag else ""
        print(
            f"{prefix}MAP refinement ({len(returned_signals)} signals): accepted={map_info['accepted']}, "
            f"+{map_info['improvement']:.3f} nats, {map_info['iterations']} iterations; {map_info['message']}\n"
            f"{prefix}  max |grad| per box width: free {map_info['free_gradient']:.2e}, "
            f"at bound {map_info['bound_gradient']:.2e} ({map_info['at_bound']} signal(s) at a box bound)\n",
            flush=True,
        )

    if return_diagnostics:
        # fs and log_Qs are (chains, draws, s), each draw already sorted by f.
        diagnostics = get_nuts_diagnostics(fs, log_Qs, extra_fields)
        return returned_signals, returned_bounds, diagnostics

    return returned_signals, returned_bounds

# Orchestrators ---------------------------------------------------------------------------

def _run_grid_search(args):
    t, d, task, existing, min_novelty, B = args

    return grid_search(
        t=t,
        d=d,
        f_min=task.f_min,
        f_max=task.f_max,
        Q_min=task.Q_min,
        Q_max=task.Q_max,
        f_points=task.f_points,
        Q_points=task.Q_points,
        existing=existing,
        min_novelty=min_novelty,
        B=B,
    )

def grid_search_orchestrator(path, signals, t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, cores, min_novelty=0.0, B=None):
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

    existing = np.asarray(signals).reshape(-1, 2)
    B = f_max - f_min if B is None else B
    args = [(t, d, task, existing, min_novelty, B) for task in tasks]

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

    if not bool(jnp.any(jnp.isfinite(log_prob_space))):
        return None

    plot_probability_surface(f"Probability Surface {len(signals)}", path / "diagnostic" / "probability_surface" / f"probability_surface_{len(signals)}.png", probability_surface)

    f_space, Q_space, log_prob_space = probability_surface
    best_index = jnp.unravel_index(jnp.argmax(log_prob_space), log_prob_space.shape)

    f_index, Q_index = best_index
    signal = (f_space[f_index], Q_space[Q_index])

    return signal

def _nuts_worker(args):
    checkpoint, t, d, B, warmup, sample, cores_per_worker, seed, tag = args
    signals, bounds = checkpoint[0], checkpoint[1]

    return nuts(
        t=t,
        d=d,
        B=B,
        signals=signals,
        bounds=bounds,
        warmup=warmup,
        sample=sample,
        chains=cores_per_worker,
        rng_key_value=seed,
        tag=tag,
    )

def nuts_orchestrator(checkpoints, t, d, B, warmup, sample, devices, tag=""):
    if not checkpoints:
        return []

    n_checkpoints = len(checkpoints)
    n_workers = min(devices, n_checkpoints)
    devices_per_worker = max(1, devices // n_checkpoints)

    tasks = [(checkpoint, t, d, B, warmup, sample, devices_per_worker, index, tag) for index, checkpoint in enumerate(checkpoints)]
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        optimized_checkpoints = list(executor.map(_nuts_worker, tasks))

    return optimized_checkpoints

def _colony_worker(args):
    ind, t_raw, d_raw, band_edges, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices_per_colony, cores_per_colony, diagnostics, signal_step, path, min_novelty, prune, fit_edge_gain, ringing_db, max_trim_seconds, filter_order, core_gain, max_condition, model_error_db = args
    n_subbands = len(band_edges) - 1
    core_min = float(band_edges[ind])
    core_max = float(band_edges[ind + 1])
    colony_path = path / f"{round(1000 * core_min, 2)}r{round(1000 * core_max, 2)}mHz_{ind + 1}r{n_subbands}"

    # Three ranges: the core [core_min, core_max] decides which signals this colony keeps; the
    # filter band is widened so the filter keeps >= core_gain of the amplitude at the core edges;
    # the fit band extends each side until the filter's gain falls to fit_edge_gain, so modes leaking
    # in from beyond the filter edges are fitted where they are. Ranges may extend past the global band.
    sampling_rate = 1 / np.mean(np.diff(t_raw))
    filter_min, filter_max, core_edge_gain = get_filter_band(core_min, core_max, sampling_rate, filter_order, core_gain)
    fit_min, fit_max = get_fit_band(filter_min, filter_max, sampling_rate, filter_order, fit_edge_gain)
    core = (core_min, core_max, ind == n_subbands - 1)
    filter_band = (filter_min, filter_max, core_edge_gain)

    signals, bounds, posterior_variance, marginal = colony(t_raw, d_raw, fit_min, fit_max, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices_per_colony, cores_per_colony, diagnostics, signal_step, colony_path, min_novelty, prune, core, ringing_db, max_trim_seconds, filter_band, filter_order, max_condition, model_error_db)

    marginal["summary"]["subband"] = ind + 1
    return ind, np.asarray(signals), np.asarray(bounds), posterior_variance, marginal

def colony_orchestrator(path, t_raw, d_raw, band_edges, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices, cores, diagnostics, signal_step, min_novelty=0.0, prune=False, fit_edge_gain=DEFAULT_FIT_EDGE_GAIN, ringing_db=DEFAULT_RINGING_DB, max_trim_seconds=DEFAULT_MAX_TRIM_SECONDS, filter_order=DEFAULT_FILTER_ORDER, core_gain=DEFAULT_CORE_GAIN, max_condition=DEFAULT_MAX_CONDITION, model_error_db=DEFAULT_MODEL_ERROR_DB):
    n_subbands = len(band_edges) - 1
    n_workers = min(devices, n_subbands)
    devices_per_colony = devices // n_workers
    cores_per_colony = cores // n_workers

    tasks = [(ind, t_raw, d_raw, band_edges, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices_per_colony, cores_per_colony, diagnostics, signal_step, path, min_novelty, prune, fit_edge_gain, ringing_db, max_trim_seconds, filter_order, core_gain, max_condition, model_error_db) for ind in range(n_subbands)]
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        results = list(executor.map(_colony_worker, tasks))

    results.sort(key=lambda result: result[0])
    signals = jnp.concatenate([jnp.asarray(result[1]).reshape(-1, 2) for result in results], axis=0)
    bounds = jnp.concatenate([jnp.asarray(result[2]).reshape(-1, 4) for result in results], axis=0)
    subband_variances = np.array([result[3] for result in results], dtype=float)
    marginals = [result[4] for result in results]

    return signals, bounds, subband_variances, marginals

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

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    threshold = jnp.finfo(g.dtype).eps ** 0.5 * jnp.max(eigenvalues)
    eigenvalues = jnp.where(eigenvalues > threshold, eigenvalues, threshold)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d
    

    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)

    return mean_sq_proj

def get_glob_ll(t, d, signals, f_min, f_max, Q_min, Q_max, B=None):
    fs, Qs = unpack_signals(signals)
    s = fs.shape[0]
    B = f_max - f_min if B is None else B   # filter bandwidth; [f_min, f_max] is the fit band
    N = int(2 * t[-1] * B)
    m = 2 * s

    if s == 0:
        residual_sq = (N / len(d)) * jnp.sum(d ** 2)
        full_rank = True
        in_bounds = True
    else:
        ks = jnp.pi * fs / Qs
        arg = (2.0 * jnp.pi * fs)[:, None] * t[None, :]
        decay = jnp.exp(-ks[:, None] * t[None, :])
        G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay)).T

        U, R = jnp.linalg.qr(G, mode="reduced")
        residual = d - U @ (U.T @ d)
        residual_sq = (N / len(d)) * jnp.sum(residual ** 2)

        diagonal = jnp.abs(jnp.diag(R))
        tolerance = jnp.finfo(G.dtype).eps * max(G.shape) * jnp.max(diagonal)
        full_rank = jnp.all(diagonal > tolerance)
        in_bounds = jnp.all((fs >= f_min) & (fs <= f_max) & (Qs >= Q_min) & (Qs <= Q_max))

    noise_variance = residual_sq / (N - m - 2.0)

    log_likelihood = -0.5 * (N * jnp.log(2.0 * jnp.pi * noise_variance) + residual_sq / noise_variance)
    score = log_likelihood - 2.0 * s * jnp.log(N)

    valid = (N > m + 2.0) & (residual_sq > 0) & (noise_variance > 0) & full_rank & in_bounds
    return jnp.where(valid, score, -jnp.inf)

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

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    threshold = jnp.finfo(g.dtype).eps ** 0.5 * jnp.max(eigenvalues)
    eigenvalues = jnp.where(eigenvalues > threshold, eigenvalues, threshold)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d
    

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)
    noise_variance = jnp.maximum(noise_variance, jnp.finfo(h.dtype).eps * sum_sq_data / (N - m - 2))
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

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    threshold = jnp.finfo(g.dtype).eps ** 0.5 * jnp.max(eigenvalues)
    eigenvalues = jnp.where(eigenvalues > threshold, eigenvalues, threshold)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)
    noise_variance = jnp.maximum(noise_variance, jnp.finfo(h.dtype).eps * sum_sq_data / (N - m - 2))

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

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    threshold = jnp.finfo(g.dtype).eps ** 0.5 * jnp.max(eigenvalues)
    eigenvalues = jnp.where(eigenvalues > threshold, eigenvalues, threshold)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
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

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    threshold = jnp.finfo(g.dtype).eps ** 0.5 * jnp.max(eigenvalues)
    eigenvalues = jnp.where(eigenvalues > threshold, eigenvalues, threshold)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d
    

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    noise_variance = (1 / (N - m - 2)) * (sum_sq_data - sum_sq_proj)
    noise_variance = jnp.maximum(noise_variance, jnp.finfo(h.dtype).eps * sum_sq_data / (N - m - 2))

    theta, unravel = ravel_pytree(signals)
    
    def objective(theta):
        return get_mean_sq_proj(t, d, B, unravel(theta))

    b = (-m / 2) * jax.hessian(objective)(theta)
    b = 0.5 * (b + b.T)

    b_eigenvalues, b_eigenvectors = jnp.linalg.eigh(b)
    threshold_b = jnp.finfo(b.dtype).eps ** 0.5 * jnp.max(b_eigenvalues)
    safe_b_eigenvalues = jnp.where(b_eigenvalues > threshold_b, b_eigenvalues, threshold_b)
    covariance = noise_variance * ((b_eigenvectors / safe_b_eigenvalues[None, :]) @ b_eigenvectors.T)
    variances = jnp.sqrt(jnp.diag(covariance))

    fs_unc = variances[::2]
    Qs_unc = variances[1::2]

    return fs_unc, Qs_unc

# Evaluation (all) ------------------------------------------------------------------------

def get_bretthorst_glob_ll(N, sum_sq_data, sum_sq_proj, rss, fisher, fs, Qs, f_min, f_max, Q_min, Q_max):
    """Bretthorst (1988) global likelihood log P(D | s signals), up to a model-independent constant.

    Built from quantities at the MAP. Priors:
      - orthonormal amplitudes ~ N(0, delta^2), delta ~ Jeffreys on a range of ratio R_delta;
      - each signal uniform in f on [f_min, f_max] and in log Q on [log Q_min, log Q_max], i.e.
        each normalized coordinate x = (x_omega, x_logQ) uniform on [-1, 1] (density 1/2);
        the s signals are exchangeable;
      - sigma ~ Jeffreys (R_sigma).
    Integrating amplitudes (prior evaluated at the projection), x (Laplace), then delta and sigma:

      log Z = [lgamma(m/2) - log 2 - log log R_delta - (m/2) log(P / 2)]          amplitudes, delta
            + sum_(lambda_i > lambda_min) [-log 2 + (1/2) log(2 pi) - (1/2) log lambda_i]   location (Occam)
            + log s!                                                              signal labellings
            + [lgamma((N-m-r)/2) - log 2 - log log R_sigma - ((N-m-r)/2) log(RSS / 2)]   sigma

    with m = 2s, P = projected energy, RSS = residual sum of squares, lambda_i the eigenvalues of
    b_x = -(1/2) Hessian of P in x coordinates (taken as the Gauss-Newton Fisher matrix
    J_perp^T J_perp, exact up to the residual term), and r = r_eff the number of determined
    directions. The location term is log(posterior width / prior width) per direction and does not
    depend on where a signal sits in the band.

    Effective rank: along an eigen-direction the posterior sd is sigma / sqrt(lambda_i), and the
    prior width is 2. The direction is determined (narrower than its prior) when
    sqrt(2 pi) sigma / sqrt(lambda_i) < 2, i.e. lambda_i > lambda_min = (pi / 2) sigma^2, with sigma^2 =
    RSS / (N - m) at the peak of the sigma integral. An undetermined direction integrates to the full
    prior mass (factor 1): it contributes 0 and brings no factor of sigma, so it is left out of r. The
    score is continuous as an eigenvalue crosses lambda_min. Nearly redundant signals therefore pay
    for their amplitudes but earn no volume from the directions the data cannot resolve.

    log s! counts the s! equivalent posterior modes (relabellings) that a single Laplace
    approximation misses. (2 pi)^(-N/2) is common to every model. All sums are raw sums over the samples; rescaling the data shifts log Z by
    -(N/2) log c for every model alike. Assumes each signal's posterior lies well inside the band.

    s = 0 (noise only): lgamma(N/2) - log 2 - log log R_sigma - (N/2) log(sum d^2 / 2).
    """
    n_signals = fs.shape[0]
    log_2 = jnp.log(2.0)
    log_log_R_sigma = jnp.log(jnp.log(BRETTHORST_RATIO_SIGMA))

    if n_signals == 0:
        return jsp.gammaln(N / 2) - log_2 - log_log_R_sigma - (N / 2) * jnp.log(sum_sq_data / 2)

    m = 2 * n_signals

    if N - m <= 0:
        return -jnp.inf

    delta_term = jsp.gammaln(m / 2) - log_2 - jnp.log(jnp.log(BRETTHORST_RATIO_DELTA)) - (m / 2) * jnp.log(sum_sq_proj / 2)
    labelling_term = jsp.gammaln(n_signals + 1.0)

    # Fisher is in (f, Q), interleaved [f1, Q1, ...]; convert to x: b_x = J^T b J with
    # J = diag(df/dx_omega, dQ/dx_logQ) = diag(omega_halfwidth / 2 pi, Q log_Q_halfwidth).
    omega_halfwidth = jnp.pi * (f_max - f_min)
    log_Q_halfwidth = 0.5 * (jnp.log(Q_max) - jnp.log(Q_min))
    jacobian = jnp.stack((jnp.full_like(fs, omega_halfwidth / (2.0 * jnp.pi)), Qs * log_Q_halfwidth), axis=-1).ravel()
    fisher_x = fisher * jnp.outer(jacobian, jacobian)
    fisher_x = 0.5 * (fisher_x + fisher_x.T)

    # Eigen-directions in x itself (no Jacobi scaling, which would rotate them off the prior box).
    eigenvalues = jnp.linalg.eigvalsh(fisher_x)
    determined = eigenvalues > 0.5 * jnp.pi * rss / (N - m)
    r_eff = int(jnp.sum(determined))
    safe_eigenvalues = jnp.where(determined, eigenvalues, 1.0)
    location_term = jnp.sum(jnp.where(determined, -log_2 + 0.5 * jnp.log(2.0 * jnp.pi) - 0.5 * jnp.log(safe_eigenvalues), 0.0))

    dof = N - m - r_eff

    if dof <= 0:
        return -jnp.inf

    sigma_term = jsp.gammaln(dof / 2) - log_2 - log_log_R_sigma - (dof / 2) * jnp.log(rss / 2)

    return delta_term + location_term + labelling_term + sigma_term

def get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=DEFAULT_MAX_CONDITION, model_error_db=DEFAULT_MODEL_ERROR_DB):
    fs, Qs = unpack_signals(signals)
    n_signals = fs.shape[0]
    m = 2 * n_signals
    N = int(2 * t[-1] * B)
    sum_sq_data = jnp.sum(d ** 2)
    safe_sum_sq_data = jnp.maximum(sum_sq_data, jnp.finfo(d.dtype).tiny)
    dof = N - m - 2
    safe_dof = jnp.maximum(dof, 1)

    if n_signals == 0:
        sum_sq_proj = jnp.asarray(0.0, dtype=d.dtype)
        rss = sum_sq_data
        fisher = jnp.empty((0, 0), dtype=d.dtype)
        residual_sq = (N / len(d)) * sum_sq_data
        full_rank = True
        in_bounds = True
        k2 = jnp.asarray(1.0, dtype=d.dtype)
        amplitudes = jnp.empty((0,), dtype=d.dtype)
        phases = jnp.empty((0,), dtype=d.dtype)
        fs_unc = jnp.empty((0,), dtype=d.dtype)
        Qs_unc = jnp.empty((0,), dtype=d.dtype)
        covariance = jnp.empty((0, 0), dtype=d.dtype)
        vif = jnp.empty((0,), dtype=d.dtype)
    else:
        z, E = get_basis(t, fs, Qs)
        G = jnp.concatenate((jnp.real(E), jnp.imag(E)))   # (m, n): [cos * decay; sin * decay]
        g = G @ G.T
        g = 0.5 * (g + g.T)

        k2 = jnp.asarray(get_basis_condition(t, jnp.stack((fs, Qs), axis=-1)), dtype=d.dtype)

        # Projection via Householder QR (backward stable; no eigendecomposition).
        G_qr = G.T
        U, R = jnp.linalg.qr(G_qr, mode="reduced")
        Ud = U.T @ d
        sum_sq_proj = jnp.sum(Ud ** 2)
        residual = d - U @ Ud
        rss = jnp.sum(residual ** 2)                    # raw RSS over len(d) samples
        residual_sq = (N / len(d)) * rss                # effective-N scaling, for BIC only
        diagonal = jnp.abs(jnp.diag(R))
        tolerance = jnp.finfo(G_qr.dtype).eps * max(G_qr.shape) * jnp.max(diagonal)
        full_rank = jnp.all(diagonal > tolerance)
        in_bounds = jnp.all((fs >= f_min) & (fs <= f_max) & (Qs >= Q_min) & (Qs <= Q_max))

        # Amplitudes: G^T beta = U R beta = U U^T d  =>  R beta = U^T d
        beta = solve_triangular(R, Ud, lower=False)
        cosine_coefficients = beta[:n_signals]
        sine_coefficients = beta[n_signals:]
        amplitudes = jnp.sqrt(cosine_coefficients ** 2 + sine_coefficients ** 2)
        phases = jnp.atan2(cosine_coefficients, sine_coefficients)

        # Variance inflation factor per basis function: VIF_i = (g^-1)_ii g_ii = ||row_i(R^-1)||^2 g_ii.
        # Per signal, report the worse of its cos/sin columns. VIF >> 10 means the signal
        # is largely explained by the others (not resolved).
        R_inv = solve_triangular(R, jnp.eye(m, dtype=R.dtype), lower=False)
        vif_basis = jnp.sum(R_inv ** 2, axis=1) * jnp.diag(g)
        vif = jnp.maximum(vif_basis[:n_signals], vif_basis[n_signals:])

        # Gauss-Newton / variable-projection Fisher matrix for (f, Q), amplitudes profiled out.
        # Model of signal j: mu_j = Re((a_j - i b_j) exp(z_j t)), so
        # d mu_j / d theta = Re((a_j - i b_j) t exp(z_j t) dz_j / d theta).
        dz_df = -jnp.pi / Qs + 2j * jnp.pi
        dz_dQ = jnp.pi * fs / Qs ** 2
        weights = (cosine_coefficients - 1j * sine_coefficients)[:, None] * t[None, :] * E
        dmu_df = jnp.real(weights * dz_df[:, None])
        dmu_dQ = jnp.real(weights * dz_dQ[:, None])
        J = jnp.stack((dmu_df, dmu_dQ), axis=1).reshape(m, -1).T   # (n, m), columns [f1, Q1, f2, Q2, ...]
        J_perp = J - U @ (U.T @ J)                                   # remove the amplitude subspace
        fisher = J_perp.T @ J_perp
        fisher = 0.5 * (fisher + fisher.T)

        # Invert in Jacobi-scaled coordinates. Fisher is PSD by construction, so a tiny
        # eigenvalue means a genuinely unidentifiable direction, not an off-mode artefact.
        fisher_scale = jnp.sqrt(jnp.maximum(jnp.diag(fisher), jnp.finfo(fisher.dtype).tiny))
        fisher_scaled = fisher / jnp.outer(fisher_scale, fisher_scale)
        f_eigenvalues, f_eigenvectors = jnp.linalg.eigh(fisher_scaled)
        f_floor = jnp.finfo(fisher.dtype).eps * jnp.maximum(jnp.max(f_eigenvalues), jnp.finfo(fisher.dtype).tiny)
        safe_f_eigenvalues = jnp.maximum(f_eigenvalues, f_floor)
        covariance_scaled = (f_eigenvectors / safe_f_eigenvalues[None, :]) @ f_eigenvectors.T

        # Posterior-consistent noise level: raw RSS over effective degrees of freedom N = 2BT.
        posterior_variance = rss / safe_dof
        covariance = posterior_variance * covariance_scaled / jnp.outer(fisher_scale, fisher_scale)
        unc = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0))
        fs_unc = unc[::2]
        Qs_unc = unc[1::2]

    noise_variance = residual_sq / safe_dof
    noise_floor = jnp.finfo(d.dtype).eps * safe_sum_sq_data / safe_dof
    noise_variance = jnp.maximum(noise_variance, noise_floor)

    ratio = jnp.clip(sum_sq_proj / safe_sum_sq_data, 0.0, 1.0 - jnp.finfo(d.dtype).eps)
    log_prob = 0.5 * (m - N) * jnp.log1p(-ratio)
    snr = jnp.where(m > 0, jnp.sqrt((m / jnp.maximum(N, 1)) * (1.0 + sum_sq_proj / (m * noise_variance))), 0.0)

    # BIC, the model-selection score, as log L_max - (k / 2) log N (higher is better; -2x this is the
    # usual BIC). Gaussian errors over N = 2BT effective samples: noise plus a model error of at least
    # sigma^2_model = 10^(model_error_db / 10) mean(d^2). Maximized over the amplitudes and over
    # sigma^2 >= sigma^2_model:
    #   sigma^2 = max(RSS_eff / N, sigma^2_model),  log L_max = -(N / 2) log(2 pi sigma^2) - RSS_eff / (2 sigma^2).
    # Above the floor this is the usual -(N / 2) (log(2 pi RSS_eff / N) + 1); below it an extra signal gains
    # only Delta RSS_eff / (2 sigma^2_model), so the 2 log N penalty wins once the fit reaches the model error.
    # k = 4 per signal (f, Q and two amplitudes); sigma is common to every model and left out.
    mean_sq_data = safe_sum_sq_data / len(d)
    model_variance = jnp.maximum(10.0 ** (model_error_db / 10.0) * mean_sq_data, jnp.finfo(d.dtype).eps * mean_sq_data)
    ml_variance = jnp.maximum(residual_sq / jnp.maximum(N, 1), model_variance)
    log_likelihood = -0.5 * N * jnp.log(2.0 * jnp.pi * ml_variance) - residual_sq / (2.0 * ml_variance)
    bic_score = log_likelihood - 2.0 * n_signals * jnp.log(jnp.maximum(N, 1))
    # A basis above max_condition is not a resolved model (a signal is a combination of the others).
    valid = (N > m + 2) & (residual_sq > 0) & (noise_variance > 0) & full_rank & in_bounds & (k2 <= max_condition)
    bic = jnp.where(valid, bic_score, -jnp.inf)

    # Bretthorst global likelihood (diagnostic only; selection uses the BIC).
    bretthorst = get_bretthorst_glob_ll(N, sum_sq_data, sum_sq_proj, rss, fisher, fs, Qs, f_min, f_max, Q_min, Q_max)
    bretthorst_valid = valid & (rss > 0) & ((n_signals == 0) | (sum_sq_proj > 0))
    glob_ll = jnp.where(bretthorst_valid & jnp.isfinite(bretthorst), bretthorst, -jnp.inf)

    return log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance, vif, bic

# Plotting --------------------------------------------------------------------------------

def save_iteration_diagnostics(path, signals, noise_variance, snr, log_prob, glob_ll, bic, k2):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostic_path / "iteration_diagnostics.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Signals", "Noise Variance", "SNR", "Probability", "Global Likelihood (Bretthorst)", "BIC", "2-norm"])
        writer.writerow([
            len(signals),
            float(noise_variance),
            float(snr),
            float(log_prob),
            float(glob_ll),
            float(bic),
            float(k2),
        ])

def save_nuts_diagnostics(path, diagnostics, fs_unc, Qs_unc):
    """Write sampler summary and per-signal posterior diagnostics.

    Rows are ordered by posterior-sorted frequency; fs_unc/Qs_unc (Fisher, at the MAP) are
    also f-sorted, so rows line up unless the MAP reorders neighbouring signals.
    """
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)

    summary_keys = ["chains", "draws_per_chain", "divergent", "divergent_fraction", "max_tree_depth_fraction", "mean_accept_prob"]

    with (diagnostic_path / "nuts_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Quantity", "Value"])
        writer.writerows((key, diagnostics[key]) for key in summary_keys)

    fs_unc = np.asarray(fs_unc)
    Qs_unc = np.asarray(Qs_unc)
    n_signals = len(diagnostics["f_mean"])

    if fs_unc.shape[0] != n_signals:
        fs_unc = np.full(n_signals, np.nan)
        Qs_unc = np.full(n_signals, np.nan)

    with (diagnostic_path / "nuts_signals.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Posterior Frequency Mean", "Posterior Frequency SD", "Bretthorst Frequency SD (full-band sigma)", "ESS Frequency", "R-hat Frequency",
            "Posterior Q Mean", "Posterior Q SD", "Bretthorst Q SD (full-band sigma)", "ESS log Q", "R-hat log Q",
        ])
        writer.writerows(zip(
            diagnostics["f_mean"], diagnostics["f_sd"], fs_unc, diagnostics["ess_f"], diagnostics["rhat_f"],
            diagnostics["Q_mean"], diagnostics["Q_sd"], Qs_unc, diagnostics["ess_log_Q"], diagnostics["rhat_log_Q"],
        ))

def print_nuts_diagnostics(diagnostics, tag=""):
    total_draws = diagnostics["chains"] * diagnostics["draws_per_chain"]
    rhat = np.concatenate((diagnostics["rhat_f"], diagnostics["rhat_log_Q"]))
    ess = np.concatenate((diagnostics["ess_f"], diagnostics["ess_log_Q"]))

    prefix = f"{tag} " if tag else ""

    print(
        f"{prefix}NUTS diagnostics:",
        f"{diagnostics['chains']} chain(s) x {diagnostics['draws_per_chain']} draws;",
        f"divergent {diagnostics['divergent']}/{total_draws} ({100 * diagnostics['divergent_fraction']:.1f}%);",
        f"max tree depth hit {100 * diagnostics['max_tree_depth_fraction']:.1f}%;",
        f"mean accept prob {diagnostics['mean_accept_prob']:.3f}",
        flush=True,
    )

    if np.any(np.isfinite(rhat)):
        print(f"{prefix}  split R-hat max {np.nanmax(rhat):.3f}; parameters with R-hat > 1.01: {int(np.sum(rhat > 1.01))}/{rhat.size}", flush=True)
    if np.any(np.isfinite(ess)):
        print(f"{prefix}  ESS min {np.nanmin(ess):.1f}; parameters with ESS < 100: {int(np.sum(ess < 100))}/{ess.size}", flush=True)

    print("", flush=True)

def save_signal_diagnostics(path, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif, extra_columns=None):
    """extra_columns: optional {header: per-signal values} appended after the standard columns."""
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostic_path / f"signals_iteration_{len(fs)}.csv"
    extra_columns = extra_columns or {}

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frequency", "Frequency Uncertainty", "Quality Factor", "Quality Factor Uncertainty", "Amplitude", "Phase", "VIF"] + list(extra_columns))
        writer.writerows(zip(fs, fs_unc, Qs, Qs_unc, amplitudes, phases, vif, *extra_columns.values()))

def empty_marginal(summary):
    """Marginal record of a colony that keeps no signals."""
    empty = np.empty(0)
    return {"fs": empty, "Qs": empty, "fs_unc": empty, "Qs_unc": empty, "amplitudes": empty, "phases": empty, "vif": empty, "covariance": np.empty((0, 0)), "summary": summary}

def make_marginal(summary, owned, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif, covariance):
    """A colony's own statistics for the signals it keeps (owned: mask over the f-sorted statistics).

    Values are taken as the colony computed them. Amplitudes and phases are on the colony's own
    clock (t = 0 at the end of its trim, summary['start_hours'] on the raw clock), which differs
    between colonies; shift_marginals_to_reference puts them on a common clock.
    """
    index = np.nonzero(np.asarray(owned))[0]
    parameter_index = interleaved_indices(index)

    return {
        "fs": np.asarray(fs)[index],
        "Qs": np.asarray(Qs)[index],
        "fs_unc": np.asarray(fs_unc)[index],
        "Qs_unc": np.asarray(Qs_unc)[index],
        "amplitudes": np.asarray(amplitudes)[index],
        "phases": np.asarray(phases)[index],
        "vif": np.asarray(vif)[index],
        "covariance": np.asarray(covariance)[np.ix_(parameter_index, parameter_index)],
        "summary": summary,
    }

def append_marginals(marginals):
    """Append colony marginal records (in subband order); the covariance is block-diagonal."""
    keys = ["fs", "Qs", "fs_unc", "Qs_unc", "amplitudes", "phases", "vif"]
    appended = {key: np.concatenate([np.asarray(marginal[key], dtype=float).reshape(-1) for marginal in marginals]) for key in keys}
    appended["subband"] = np.concatenate([np.full(len(marginal["fs"]), marginal["summary"]["subband"], dtype=int) for marginal in marginals])

    size = 2 * len(appended["fs"])
    covariance = np.zeros((size, size))
    position = 0
    for marginal in marginals:
        block = np.asarray(marginal["covariance"])
        covariance[position:position + block.shape[0], position:position + block.shape[0]] = block
        position += block.shape[0]
    appended["covariance"] = covariance

    return appended

def shift_marginals_to_reference(marginals, reference_hours=None):
    """Express every colony's amplitudes and phases at one common time.

    Each colony fits A exp(-pi f t / Q) sin(2 pi f t + phi) with t = 0 at the end of its own trim
    (start_hours on the raw clock). At a later reference time, Delta = reference - start >= 0:

        A_ref = A exp(-pi f Delta / Q),   phi_ref = phi + 2 pi f Delta  (wrapped to (-pi, pi]),

    the same signal written on the shared clock. The default reference is the latest start, i.e.
    the subband that trimmed the most, so every colony's record covers it. Frequencies, Qs and their
    covariance are unchanged. Modifies the marginals in place and returns the reference (hours).
    """
    if reference_hours is None:
        reference_hours = max(float(marginal["summary"]["start_hours"]) for marginal in marginals)
    for marginal in marginals:
        delta = (reference_hours - float(marginal["summary"]["start_hours"])) * 3600.0
        fs = np.asarray(marginal["fs"], dtype=float)
        Qs = np.asarray(marginal["Qs"], dtype=float)
        marginal["amplitudes"] = np.asarray(marginal["amplitudes"], dtype=float) * np.exp(-np.pi * fs * delta / Qs)
        marginal["phases"] = np.angle(np.exp(1j * (np.asarray(marginal["phases"], dtype=float) + 2.0 * np.pi * fs * delta)))
        marginal["summary"]["reference_hours"] = reference_hours
    return reference_hours

def save_subband_summary(path, summaries):
    diagnostic_path = Path(path)
    diagnostic_path.mkdir(parents=True, exist_ok=True)
    columns = [
        ("Subband", "subband"), ("Core Min (Hz)", "core_min"), ("Core Max (Hz)", "core_max"),
        ("Filter Min (Hz)", "filter_min"), ("Filter Max (Hz)", "filter_max"), ("Fit Min (Hz)", "fit_min"), ("Fit Max (Hz)", "fit_max"),
        ("Amplitude Kept at Core Edges", "core_edge_gain"), ("Trim (h)", "trim_hours"), ("Record Start (h)", "start_hours"),
        ("Reference Time (h)", "reference_hours"), ("Selected Signals", "selected"),
        ("Kept Signals", "kept"), ("Global Likelihood (Bretthorst)", "glob_ll"), ("BIC", "bic"),
        ("Noise Scale v", "noise_scale"),
    ]

    with (diagnostic_path / "subband_summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([header for header, _ in columns])
        for summary in summaries:
            writer.writerow([summary.get(key, "") for _, key in columns])

def plot_probability_surface(title, path, probability_surface):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f_space, Q_space, log_prob_space = probability_surface
    # Unresolvable (novelty-masked) candidates are -inf; leave them blank.
    log_prob_space = np.ma.masked_invalid(np.asarray(log_prob_space).T)

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

def plot_signal_space(title, path, fs, Qs, fs_unc, Qs_unc, core_edges=None, f_range=None, edges_label="Core band edges", filter_edges=None):
    """core_edges: frequencies drawn as vertical lines (a colony's core band, or the subband
    boundaries); the region of f_range outside the outermost core edges is shaded as padding."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fs = np.asarray(fs)
    Qs = np.asarray(Qs)
    fs_unc = np.asarray(fs_unc)
    Qs_unc = np.asarray(Qs_unc)

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_yscale("log")
    ax.scatter(fs, Qs, color="red", s=10, zorder=3, label="Signals")

    signal_ylim = ax.get_ylim()

    for f, q, f_unc, q_unc in zip(fs, Qs, fs_unc, Qs_unc):
        for sigma, alpha in ((3.0, 0.06), (2.0, 0.12), (1.0, 0.22)):
            ellipse = Ellipse(xy=(f, q), width=2.0 * sigma * f_unc, height=2.0 * sigma * q_unc, facecolor="red", edgecolor="red", alpha=alpha, linewidth=0.8)
            ax.add_patch(ellipse)

    ax.set_ylim(signal_ylim)

    if f_range is not None:
        ax.set_xlim(*f_range)

    draw_core_edges(ax, core_edges, edges_label, filter_edges)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Quality Factor (Q)")
    ax.set_title(title)

    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

    return

def plot_time_series(title, path, t, d, signals=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if signals is not None:
        fig, (ax, ax_resid) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

        model = get_model(t, d, signals)

        ax.plot(t, d, color="black", label="Data")
        ax.plot(t, model, color="red", label="Model", alpha=0.7)
        ax.set_ylabel("Intensity")
        ax.legend()

        ax_resid.plot(t, d - model, color="blue", linewidth=0.9, label="Residual")
        ax_resid.axhline(0, color="gray", linewidth=0.8)
        ax_resid.set_ylabel("Residual")
        ax_resid.set_xlabel("Time (s)")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(t, d, color="black", label="Data")
        ax.set_ylabel("Intensity")
        ax.set_xlabel("Time (s)")

    ax.set_xlim(t[0], t[-1])
    ax.set_title(title)

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

def draw_core_edges(ax, core_edges, label="Core band edges", filter_edges=None):
    """Solid lines at the core-band edges with everything outside them shaded (not kept), and
    dashed lines at the filter-band edges when given."""
    if core_edges is None or len(core_edges) == 0:
        return

    edges = np.sort(np.asarray(core_edges, dtype=float))
    x_min, x_max = ax.get_xlim()

    if x_min < edges[0]:
        ax.axvspan(x_min, edges[0], color="grey", alpha=0.12, linewidth=0, label="Outside core (not kept)")
    if x_max > edges[-1]:
        ax.axvspan(edges[-1], x_max, color="grey", alpha=0.12, linewidth=0, label="Outside core (not kept)" if x_min >= edges[0] else None)

    for index, edge in enumerate(edges):
        ax.axvline(edge, color="green", linestyle="-", linewidth=0.9, alpha=0.8, label=label if index == 0 else None)

    if filter_edges is not None:
        for index, edge in enumerate(filter_edges):
            ax.axvline(edge, color="dimgrey", linestyle="--", linewidth=0.9, alpha=0.9, label="Filter band edges" if index == 0 else None)

    ax.set_xlim(x_min, x_max)
    ax.legend()

def plot_fourier_space(title, path, t, d, f_min, f_max, signals=None, edges_init=None, band_edges=None, core_edges=None, edges_label="Core band edges", filter_edges=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    f, p = get_fft(t, d, f_min, f_max)
    outputs = [(path, None)]

    if signals is not None:
        model = get_model(t, d, signals)
        fs, Qs = unpack_signals(signals)
        ks = jnp.pi * fs / Qs
        amplitudes, phases = get_phasor(t, d, signals)

        lorentzian_p = get_model_fourier(f, fs, ks, amplitudes, phases)
        _, fft_p = get_fft(t, model, f_min, f_max)

        outputs = [
            (path.with_name(f"fourier_space_{len(signals)}_lorentzian{path.suffix}"), lorentzian_p),
            (path.with_name(f"fourier_space_{len(signals)}_fft{path.suffix}"), fft_p),
        ]

    for output_path, model_p in outputs:
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(f, abs(p), color="black", label="Data", linewidth=0.9)
        ax.set_xlim(f[0], f[-1])
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)

        if model_p is not None:
            ax.plot(f, abs(model_p), color="red", alpha=0.8, label="Model", linewidth=0.9)
            residual = abs(p) - abs(model_p)
            ax.plot(f, residual, color="blue", alpha=0.6, label="Residual", linewidth=0.9)

            ymin, ymax = ax.get_ylim()
            ymin_val = -jnp.abs(jnp.min(residual)) * 100
            ymax_val = jnp.abs(jnp.max(residual)) * 100
            ax.vlines(fs, ymin_val, ymax_val, color="red", linestyle="--", alpha=0.15, linewidth=0.4)
            ax.set_ylim(ymin, ymax)
            ax.legend()

        if edges_init is not None:
            for edge in edges_init:
                ax.axvline(edge, color="red", linestyle="--")

        if band_edges is not None:
            for edge in band_edges:
                ax.axvline(edge, color="blue", linestyle="--")

        draw_core_edges(ax, core_edges, edges_label, filter_edges)

        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)

SUBBAND_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

def plot_filter_responses(path, band_edges, f_min, f_max, sampling_rate, filter_order, core_gain, fit_edge_gain):
    """Zero-phase amplitude gain |H|^2 of every subband filter over the full band, linear and in dB.

    Solid grey lines: core (subband) boundaries. Dotted lines: the core_gain target (linear panel)
    and the fit_edge_gain target (dB panel), where each fit band ends. Colored
    brackets above the curves: each subband's fit band. Dashed black: the full-band filter.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    band_edges = np.asarray(band_edges, dtype=float)
    n_subbands = len(band_edges) - 1

    bands = []
    for ind in range(n_subbands):
        core_min, core_max = float(band_edges[ind]), float(band_edges[ind + 1])
        filter_min, filter_max, edge_gain = get_filter_band(core_min, core_max, sampling_rate, filter_order, core_gain)
        fit_min, fit_max = get_fit_band(filter_min, filter_max, sampling_rate, filter_order, fit_edge_gain)
        bands.append((core_min, core_max, filter_min, filter_max, fit_min, fit_max, edge_gain))

    full_filter_min, full_filter_max, _ = get_filter_band(f_min, f_max, sampling_rate, filter_order, core_gain)
    x_min = 0.9 * min(band[4] for band in bands)
    x_max = 1.05 * max(band[5] for band in bands)
    frequencies = np.linspace(x_min, x_max, 6000)

    def gain(low, high):
        sos = butter(filter_order, [low, high], btype="bandpass", fs=sampling_rate, output="sos")
        _, response = sosfreqz(sos, worN=frequencies, fs=sampling_rate)
        return np.abs(response) ** 2

    fig, (ax_linear, ax_db) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    mhz = frequencies * 1e3

    for ind, (core_min, core_max, filter_min, filter_max, fit_min, fit_max, edge_gain) in enumerate(bands):
        color = SUBBAND_COLORS[ind % len(SUBBAND_COLORS)]
        response = gain(filter_min, filter_max)
        label = f"Subband {ind + 1}: core {1e3 * core_min:.2f}-{1e3 * core_max:.2f} mHz ({edge_gain:.3f} at core edges)"
        ax_linear.plot(mhz, response, color=color, linewidth=2, label=label)
        ax_db.plot(mhz, 20 * np.log10(np.maximum(response, 1e-12)), color=color, linewidth=2)
        bracket_y = 1.10 + 0.05 * (ind % 2)
        ax_linear.plot([1e3 * fit_min, 1e3 * fit_max], [bracket_y, bracket_y], color=color, linewidth=2, solid_capstyle="butt")
        for x in (fit_min, fit_max):
            ax_linear.plot([1e3 * x, 1e3 * x], [bracket_y - 0.02, bracket_y], color=color, linewidth=2)

    full_response = gain(full_filter_min, full_filter_max)
    ax_linear.plot(mhz, full_response, color="black", linestyle="--", linewidth=1.2, label="Full-band filter")
    ax_db.plot(mhz, 20 * np.log10(np.maximum(full_response, 1e-12)), color="black", linestyle="--", linewidth=1.2)

    for ax in (ax_linear, ax_db):
        for index, edge in enumerate(band_edges):
            ax.axvline(1e3 * edge, color="#52514e", linewidth=0.8, alpha=0.6, label="Core boundaries" if (ax is ax_linear and index == 0) else None)
        ax.grid(False)

    ax_linear.axhline(core_gain, color="#52514e", linestyle=":", linewidth=1.0, label=f"Target gain at core edges ({core_gain:g})")
    ax_linear.text(1e3 * x_min, 1.175, "  fit bands", color="#52514e", va="bottom", fontsize=9)
    ax_linear.set_ylim(0, 1.22)
    ax_linear.set_ylabel("Amplitude gain |H|^2 (zero-phase)")
    ax_linear.set_title(f"Subband filter responses (Butterworth order {filter_order}, applied forward and backward)")

    ax_db.axhline(20 * np.log10(fit_edge_gain), color="#52514e", linestyle=":", linewidth=1.0)
    ax_db.set_ylim(-100, 5)
    ax_db.set_xlim(1e3 * x_min, 1e3 * x_max)
    ax_db.set_xlabel("Frequency (mHz)")
    ax_db.set_ylabel("Amplitude gain (dB)")

    handles, labels = ax_linear.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)

def initial_plot(path, t, d, f_min, f_max, core_edges=None, filter_edges=None):
    plot_time_series("Time Series", path / "time_series.png", t, d)
    plot_fourier_space("Fourier Space", path / "fourier_space.png", t, d, f_min, f_max, core_edges=core_edges, filter_edges=filter_edges)
    
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

def get_bounds(signals, f_min, f_max, Q_min, Q_max, f_margin=1 / 8, log_Q_margin=1 / 8):
    fs, Qs = unpack_signals(signals)

    log_Q_space_min = jnp.log(Q_min)
    log_Q_space_max = jnp.log(Q_max)

    frequency_min = jnp.maximum(fs - f_margin, f_min)
    frequency_max = jnp.minimum(fs + f_margin, f_max)

    log_Qs = jnp.log(Qs)
    log_Q_min = jnp.maximum(log_Qs - log_Q_margin, log_Q_space_min)
    log_Q_max = jnp.minimum(log_Qs + log_Q_margin, log_Q_space_max)

    quality_factor_min = jnp.exp(log_Q_min)
    quality_factor_max = jnp.exp(log_Q_max)

    return jnp.stack((frequency_min, frequency_max, quality_factor_min, quality_factor_max), axis=-1)

def get_shift(x, y):
    a, b, c = np.polyfit(x, y, 2)
    x_shift = -b / (2 * a)

    return x_shift

def get_band_edges(path, t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, subbands):
    bandwidth = (f_max - f_min) / subbands
    edges = np.empty(subbands + 1)
    edges_init = np.empty(subbands + 1)
    edges[0] = f_min
    edges[-1] = f_max
    edges_init[0] = f_min
    edges_init[-1] = f_max

    for i in range(subbands - 1):
        f_edge_init = f_min + (i + 1) * bandwidth
        half_width = bandwidth / 4

        f_space, Q_space, log_prob_space = grid_search(t, d, f_edge_init - half_width, f_edge_init + half_width, Q_min, Q_max, f_points, Q_points)

        shifted = log_prob_space - np.min(log_prob_space)
        max_shifted = np.max(shifted)
        normalized = shifted / max_shifted if max_shifted > 0 else np.zeros_like(shifted)

        penalty = ((f_space - f_edge_init) / half_width) ** 2
        weighted = normalized # + penalty[:, None]

        frequency_scores = np.min(weighted, axis=1)
        best_f_idx = np.argmin(frequency_scores)
        edges[i + 1] = f_space[best_f_idx]
        edges_init[i + 1] = f_edge_init

    return edges_init, edges

def load_data(type, event, network, station):
    if type == "observed":
        path = Path.cwd() / type / event / f"{network}_{station}_TS.ascii"
    elif type in ("synthetic", "prem"):
        path = Path.cwd() / type / event / f"{network}_{station}.dat"
    else:
        raise ValueError(f"Unknown data type {type!r}; expected 'observed', 'synthetic' or 'prem'.")
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]

def get_filter_time_constant(fs, f_min, f_max, order=DEFAULT_FILTER_ORDER):
    """Decay time (s) of the filter's slowest pole, tau = -1 / (fs ln|p|): how long ringing lasts.

    For a narrow Butterworth bandpass the slowest poles sit near the band edges, so tau is set by
    the bandwidth (roughly 1 / (2 pi (B / 2) sin(pi / (2 order)))); for a wide band it is set by
    the lower edge frequency instead. Measured from the actual poles, so both cases are covered.
    """
    _, poles, _ = butter(order, [f_min, f_max], btype="bandpass", fs=fs, output="zpk")
    decay_rates = -np.log(np.abs(poles)) * fs
    return 1.0 / np.min(decay_rates)

def get_ringing_envelope(sos, fs, time_constant):
    """Ringing envelope of the forward-backward filter, sample by sample after an impulse.

    Measured directly on sosfiltfilt (whose doubled poles decay like (1 + t/tau) exp(-t/tau),
    slower than exp(-t/tau)): filter a unit impulse placed mid-way through 40 tau of zeros and
    normalize |response| by its peak. Entry k is the largest |response| at any lag >= k samples,
    so the envelope is monotone and "below a level from here on" is a single comparison.
    -60 dB is reached after about 5-6 tau.
    """
    half_length = int(20 * time_constant * fs) + 1
    impulse = np.zeros(2 * half_length + 1)
    impulse[half_length] = 1.0
    response = np.abs(sosfiltfilt(sos, impulse))
    response /= response.max()
    tail = response[half_length:]
    return np.maximum.accumulate(tail[::-1])[::-1]

def get_fit_band(filter_min, filter_max, fs, order=DEFAULT_FILTER_ORDER, fit_edge_gain=DEFAULT_FIT_EDGE_GAIN):
    """Fit band around a filter band: each side extends outward from the filter edge until the
    zero-phase (forward-backward) amplitude gain |H(f)|^2 falls to fit_edge_gain, solved per side by
    root-finding (the skirt is monotonic outside the passband). A mode beyond the fit band therefore
    leaks in at no more than fit_edge_gain of its amplitude, and modes inside it are fitted where they
    are. The upper side reaches further, since the skirt falls more slowly above the band in linear
    frequency. The lower edge stops at half the filter's lower edge and the upper edge at 0.45 fs;
    at a limit the gain there is above the target. fit_edge_gain >= 1/2 gives the filter band itself.
    Returns (fit_min, fit_max).
    """
    if fit_edge_gain >= 0.5:
        return filter_min, filter_max
    sos = butter(order, [filter_min, filter_max], btype="bandpass", fs=fs, output="sos")

    def excess(f):
        _, response = sosfreqz(sos, worN=[f], fs=fs)
        return float(np.abs(response[0]) ** 2) - fit_edge_gain

    lower_limit, upper_limit = 0.5 * filter_min, 0.45 * fs
    fit_min = lower_limit if excess(lower_limit) > 0 else brentq(excess, lower_limit, filter_min, xtol=1e-12)
    fit_max = upper_limit if excess(upper_limit) > 0 else brentq(excess, filter_max, upper_limit, xtol=1e-12)
    return fit_min, fit_max

def get_filter_band(core_min, core_max, fs, order=DEFAULT_FILTER_ORDER, core_gain=DEFAULT_CORE_GAIN):
    """Filter passband that keeps at least core_gain of a signal's amplitude at both core edges.

    The zero-phase (forward-backward) filter's amplitude gain is |H(f)|^2, which is 1/2 (-6 dB) at
    its own edges. Each side of the passband is widened by the smallest amount that brings the gain
    at its core edge up to core_gain, solved per side by root-finding and iterated because the two
    sides interact slightly. The lower edge never goes below half the core's lower edge, and the
    upper edge stays below 0.45 fs; if a limit stops a side short, the achieved gain is returned.
    For order 8 and core_gain 0.99, each side moves out by roughly 0.2 core widths.
    Returns (filter_min, filter_max, gain achieved at the weaker core edge).
    """
    width = core_max - core_min
    lower_limit = 0.5 * core_min
    upper_limit = 0.45 * fs

    def edge_gains(low, high):
        sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
        _, response = sosfreqz(sos, worN=[core_min, core_max], fs=fs)
        return np.abs(response) ** 2

    def smallest_widening(deficit, k_limit):
        """Smallest k in [0, k_limit] with deficit(k) >= 0, or k_limit if none."""
        if deficit(0.0) >= 0:
            return 0.0
        k_high = min(0.25, k_limit)
        while deficit(k_high) < 0 and k_high < k_limit:
            k_high = min(2 * k_high, k_limit)
        if deficit(k_high) < 0:
            return k_limit
        return brentq(deficit, 0.0, k_high, xtol=1e-7)

    k_lower_limit = (core_min - lower_limit) / width
    k_upper_limit = (upper_limit - core_max) / width
    low, high = core_min, core_max

    for _ in range(20):
        k_upper = smallest_widening(lambda k: edge_gains(low, core_max + k * width)[1] - core_gain, k_upper_limit)
        new_high = core_max + k_upper * width
        k_lower = smallest_widening(lambda k: edge_gains(core_min - k * width, new_high)[0] - core_gain, k_lower_limit)
        new_low = core_min - k_lower * width
        converged = abs(new_low - low) < 1e-9 * width and abs(new_high - high) < 1e-9 * width
        low, high = new_low, new_high
        if converged:
            break

    achieved = float(np.min(edge_gains(low, high)))
    if achieved < core_gain - 1e-6:
        print(f"Warning: filter for {1000 * core_min:.3f}-{1000 * core_max:.3f} mHz keeps only {achieved:.3f} of the amplitude at the core edges.", flush=True)

    return low, high, achieved

def filter_data(t, d, f_min, f_max, order=DEFAULT_FILTER_ORDER, ringing_db=DEFAULT_RINGING_DB, max_trim_seconds=DEFAULT_MAX_TRIM_SECONDS, return_info=False):
    """Zero-phase bandpass, then trim both ends.

    Trim = min(max_trim_seconds, max(MIN_TRIM_SECONDS, time for ringing to fall below ringing_db)).
    With return_info, also returns a dict: trim (s), time constant (s), the ringing level left at
    the trim (dB), and whether the cap was the binding limit.
    """
    fs = 1 / np.mean(np.diff(t))
    sos = butter(order, [f_min, f_max], btype="bandpass", fs=fs, output="sos")
    filtered_d = sosfiltfilt(sos, d)

    time_constant = get_filter_time_constant(fs, f_min, f_max, order)
    envelope = get_ringing_envelope(sos, fs, time_constant)
    below = np.nonzero(envelope <= 10 ** (ringing_db / 20))[0]
    ringing_seconds = below[0] / fs if len(below) else len(envelope) / fs

    trim_seconds = min(max_trim_seconds, max(MIN_TRIM_SECONDS, ringing_seconds))
    trim = int(np.ceil(trim_seconds * fs))
    level_at_trim = envelope[min(trim, len(envelope) - 1)]
    ringing_left_db = 20 * np.log10(max(level_at_trim, 1e-300))

    if 2 * trim >= len(t):
        raise ValueError(
            f"Filter {1000 * f_min:.3f}-{1000 * f_max:.3f} mHz needs {trim_seconds / 3600:.1f} h trimmed "
            f"at each end (ringing to {ringing_db:.0f} dB), longer than half the "
            f"{(t[-1] - t[0]) / 3600:.1f} h record. Widen the subband or raise ringing_db."
        )

    t_trimmed = t[trim:-trim]
    result = (t_trimmed - t_trimmed[0], filtered_d[trim:-trim])

    if return_info:
        info = {
            "trim_seconds": trim_seconds,
            "start_seconds": float(t[trim] - t[0]),   # trimmed record's t = 0 on the raw clock
            "time_constant": time_constant,
            "ringing_left_db": ringing_left_db,
            "capped": ringing_seconds > max_trim_seconds,
        }
        return result + (info,)

    return result

# Run -------------------------------------------------------------------------------------

def colony(t, d, f_min, f_max, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices, cores, diagnostics, signal_step, path, min_novelty=0.0, prune=False, core=None, ringing_db=DEFAULT_RINGING_DB, max_trim_seconds=DEFAULT_MAX_TRIM_SECONDS, filter_band=None, filter_order=DEFAULT_FILTER_ORDER, max_condition=DEFAULT_MAX_CONDITION, model_error_db=DEFAULT_MODEL_ERROR_DB):
    """Fit one subband, using three nested frequency ranges.

    core = (core_min, core_max, closed_above): the true subband; only signals whose frequency lies
        in it are returned. None = keep everything in the fit band.
    filter_band = (filter_min, filter_max, core_edge_gain): the band the data are filtered to, wide
        enough that the zero-phase filter keeps >= core_gain of a signal's amplitude at the core
        edges. It sets B (N = 2BT), because it is the band the noise passes through. None = the core.
    [f_min, f_max]: the fit band, the filter band plus padding. Signals are searched and fitted
        here: a damped sinusoid passes the filter with only its amplitude changed, so modes leaking
        in from beyond the filter edges are fitted at their true f and Q.
    """
    core_min, core_max, closed_above = core if core is not None else (f_min, f_max, True)
    filter_min, filter_max, core_edge_gain = filter_band if filter_band is not None else (core_min, core_max, 0.5)

    t, d, filter_info = filter_data(t, d, filter_min, filter_max, order=filter_order, ringing_db=ringing_db, max_trim_seconds=max_trim_seconds, return_info=True)
    core_edges = (core_min, core_max)
    filter_edges = (filter_min, filter_max)

    B = filter_max - filter_min
    colony_summary = {
        "core_min": core_min, "core_max": core_max, "filter_min": filter_min, "filter_max": filter_max,
        "fit_min": f_min, "fit_max": f_max, "core_edge_gain": core_edge_gain, "trim_hours": filter_info["trim_seconds"] / 3600,
        "start_hours": filter_info["start_seconds"] / 3600,
    }

    initial_plot(path, t, d, f_min, f_max, core_edges, filter_edges)

    residual = d

    thumbprints = []
    checkpoints = [None] * depth
    signals = jnp.empty((0, 2))
    bounds = jnp.empty((0, 4))

    log_prob, glob_ll, noise_variance, snr, k2, *_, bic = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)
    save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, bic, k2)

    # The s = 0 score is the first measured point: the BIC is assumed unimodal in the signal count,
    # so the greedy search stops as soon as a refined score falls below the previous one.
    zero_glob_ll = float(glob_ll)
    thumbprints.append((0, float(bic)))

    start_time = perf_counter()
    tag = f"[{1000 * core_min:.3f}-{1000 * core_max:.3f} mHz]"
    if filter_info["capped"]:
        trim_note = f"capped at {max_trim_seconds / 3600:.0f} h; ringing left at {filter_info['ringing_left_db']:.0f} dB (target {ringing_db:.0f} dB)"
    elif filter_info["trim_seconds"] <= MIN_TRIM_SECONDS:
        trim_note = f"{MIN_TRIM_SECONDS / 3600:.0f} h minimum; ringing left at {filter_info['ringing_left_db']:.0f} dB"
    else:
        trim_note = f"ringing below {ringing_db:.0f} dB"

    print(
        f"{tag} Starting... (BIC at 0 signals: {float(bic):.2f}; Bretthorst {float(glob_ll):.2f}; max condition number {max_condition:g}; model error {model_error_db:g} dB)\n"
        f"{tag} Bands: core {1000 * core_min:.3f}-{1000 * core_max:.3f} mHz; filter {1000 * filter_min:.3f}-{1000 * filter_max:.3f} mHz "
        f"(order {filter_order}, {core_edge_gain:.3f} of amplitude kept at the core edges); fit {1000 * f_min:.3f}-{1000 * f_max:.3f} mHz\n"
        f"{tag} Filter time constant {filter_info['time_constant'] / 3600:.2f} h; trimmed {filter_info['trim_seconds'] / 3600:.1f} h "
        f"at each end ({trim_note}); {t[-1] / 3600:.1f} h of record used\n",
        flush=True,
    )

    for iteration in range(depth):
        iteration_start = perf_counter()

        if len(signals) > 0:
            model = get_model(t, d, signals)
            residual = d - model

        grid_start = perf_counter()
        signal = grid_search_orchestrator(path, signals, t, residual, f_min, f_max, Q_min, Q_max, f_points, Q_points, cores, min_novelty, B)
        grid_elapsed = perf_counter() - grid_start

        if signal is None:
            print(f"{tag} No resolvable candidate (novelty >= {min_novelty}) left; stopping greedy search at {len(signals)} signals.\n", flush=True)
            break

        signal_bounds = get_bounds(signal, f_min, f_max, Q_min, Q_max, f_margin, log_Q_margin)

        signal = jnp.asarray(signal).reshape(1, 2)
        signal_bounds = jnp.asarray(signal_bounds).reshape(1, 4)

        signals = jnp.concatenate([signals, signal], axis=0)
        bounds = jnp.concatenate([bounds, signal_bounds], axis=0)

        refined = len(signals) % signal_step == 0

        nuts_start = perf_counter()
        if refined:
            signals, bounds = nuts(t, d, B, signals, bounds, warmup, sample, devices, tag=tag)
        nuts_elapsed = perf_counter() - nuts_start

        # Linear dependence: once the newest signal is (numerically) a combination of the others, all
        # resolvable signals are taken to be fitted. Discard this model and stop; the previous count is
        # then the last checkpoint and is refined below if it has not been already.
        condition = get_basis_condition(t, signals)
        if condition > max_condition:
            print(
                f"{tag} Basis condition number {condition:.3g} exceeds {max_condition:g} at {len(signals)} signals "
                f"(signal {len(signals)} is not linearly independent of the others); stopping greedy search at {len(signals) - 1} signals.\n",
                flush=True,
            )
            break

        if diagnostics == "full":
            checkpoints[iteration] = (signals, bounds)

            log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance, vif, bic = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)

            if refined:
                thumbprints.append((len(signals), float(bic)))

            save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, bic, k2)
            save_signal_diagnostics(path / "diagnostic" / "tables", fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif)
            plot_time_series("Time Series", path / "diagnostic" / "time_series" /f"time_series_{len(signals)}.png", t, d, signals)
            plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals, core_edges=core_edges, filter_edges=filter_edges)
            plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc, core_edges=core_edges, f_range=(f_min, f_max), filter_edges=filter_edges)
            plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "diagnostic" / "covariance" / f"covariance_{len(signals)}.png", signals, covariance)
        else:
            checkpoints[iteration] = (signals, bounds)

            if refined:
                log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance, vif, bic = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)
                thumbprints.append((len(signals), float(bic)))

                save_iteration_diagnostics(path / "diagnostic" / "tables", signals, noise_variance, snr, log_prob, glob_ll, bic, k2)
                save_signal_diagnostics(path / "diagnostic" / "tables", fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif)
                plot_time_series("Time Series", path / "diagnostic" / "time_series" / f"time_series_{len(signals)}.png", t, d, signals)
                plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "diagnostic" / "fourier_space" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals, core_edges=core_edges, filter_edges=filter_edges)
                plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "diagnostic" / "signal_space" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc, core_edges=core_edges, f_range=(f_min, f_max), filter_edges=filter_edges)
                plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "diagnostic" / "covariance" / f"covariance_{len(signals)}.png", signals, covariance)

        iteration_elapsed = perf_counter() - iteration_start
        total_elapsed = perf_counter() - start_time

        likelihood_line = f"{tag} BIC at {len(signals)} signals: {thumbprints[-1][1]:.2f} (Bretthorst {float(glob_ll):.2f})\n" if refined else ""

        print(
            f"{tag} Signal {len(signals)} completed in {iteration_elapsed:.2f} seconds (total elapsed: {total_elapsed:.2f} seconds)\n"
            f"{tag} Grid search time: {grid_elapsed:.2f} seconds; NUTS time: {nuts_elapsed:.2f} seconds\n"
            f"{likelihood_line}",
            flush=True,
        )

        # Early stop: only on a refined score lower than the previous refined score.
        if refined and thumbprints[-1][1] < thumbprints[-2][1]:
            (previous_count, previous_score), (current_count, current_score) = thumbprints[-2], thumbprints[-1]
            print(
                f"{tag} BIC decreased from {previous_score:.2f} ({previous_count} signals) "
                f"to {current_score:.2f} ({current_count} signals); stopping greedy search.\n",
                flush=True,
            )
            break

    final_start = perf_counter()
    checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint is not None]

    if not checkpoints:
        print(f"{tag} Colony found no resolvable signals.\n", flush=True)
        noise_scale = get_posterior_variance(t, d, B, jnp.empty((0, 2)))
        colony_summary.update(selected=0, kept=0, glob_ll=zero_glob_ll, bic=thumbprints[0][1], noise_scale=noise_scale)
        return jnp.empty((0, 2)), jnp.empty((0, 4)), noise_scale, empty_marginal(colony_summary)

    checkpoint_by_count = {int(checkpoint[0].shape[0]): checkpoint for checkpoint in checkpoints}
    last_count = max(checkpoint_by_count)
    checkpoint_by_count[0] = (jnp.empty((0, 2)), jnp.empty((0, 4)))

    # If the greedy search stopped early (or depth is not a multiple of signal_step), the last
    # checkpoint was never refined. Refine and score it so the bisection has a right-hand anchor
    # scored the same way as every other count (at the MAP).
    if last_count not in {int(count) for count, _ in thumbprints}:
        last_signals, last_bounds = checkpoint_by_count[last_count][0], checkpoint_by_count[last_count][1]
        checkpoint_by_count[last_count] = nuts(t, d, B, last_signals, last_bounds, warmup, sample, devices, tag=tag)
        last_bic = float(get_statistics(t, d, B, checkpoint_by_count[last_count][0], f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)[-1])
        print(f"{tag} Refined the last checkpoint: BIC at {last_count} signals: {last_bic:.2f}\n", flush=True)
        thumbprints.append((last_count, last_bic))

    thumbprints = np.asarray(thumbprints, dtype=float)

    scores = {int(count): float(score) for count, score in thumbprints if np.isfinite(score) and int(count) in checkpoint_by_count}
    refined_checkpoints = {count: checkpoint_by_count[count] for count in scores}

    while True:
        best_count = max(scores, key=scores.get)
        measured_counts = sorted(scores)

        left_counts = [count for count in measured_counts if count < best_count]
        right_counts = [count for count in measured_counts if count > best_count]

        left_limit = left_counts[-1] if left_counts else 0
        right_limit = right_counts[0] if right_counts else last_count

        candidates = []

        if best_count - left_limit > 1:
            candidate = (left_limit + best_count) // 2
            if candidate not in scores and candidate in checkpoint_by_count:
                candidates.append(candidate)

        if right_limit - best_count > 1:
            candidate = (best_count + right_limit) // 2
            if candidate not in scores and candidate in checkpoint_by_count:
                candidates.append(candidate)

        candidates = sorted(set(candidates))
        if not candidates:
            break

        candidate_refits = nuts_orchestrator([checkpoint_by_count[count] for count in candidates], t, d, B, warmup, sample, devices, tag=tag)

        for count, checkpoint in zip(candidates, candidate_refits):
            refined_checkpoints[count] = checkpoint
            statistics = get_statistics(t, d, B, checkpoint[0], f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)

            log_prob, glob_ll, noise_variance, snr, k2 = statistics[:5]
            bic = statistics[-1]
            scores[count] = float(bic)
            print(f"{tag} Bisection: BIC at {count} signals: {float(bic):.2f} (Bretthorst {float(glob_ll):.2f})\n", flush=True)

            save_iteration_diagnostics(path / "result", checkpoint[0], noise_variance, snr, log_prob, glob_ll, bic, k2)

    best_count = max(scores, key=scores.get)
    signals, bounds = refined_checkpoints[best_count]
    print(f"{tag} Selected {best_count} signals (BIC {scores[best_count]:.2f})\n", flush=True)

    if best_count == 0:
        print(f"{tag} Colony completed in {perf_counter() - start_time:.2f} seconds with 0 signals\n", flush=True)
        noise_scale = get_posterior_variance(t, d, B, signals)
        colony_summary.update(selected=0, kept=0, glob_ll=zero_glob_ll, bic=scores[0], noise_scale=noise_scale)
        return signals, bounds, noise_scale, empty_marginal(colony_summary)

    if prune:
        signals, bounds, removed = prune_signals(t, d, B, signals, bounds, f_min, f_max, Q_min, Q_max, max_condition, model_error_db)
        removed_lines = "".join(f"\n{tag}   removed f={f_removed:.6e} Hz, Q={Q_removed:.2f}" for f_removed, Q_removed in removed)
        print(f"{tag} Pruning removed {len(removed)} signal(s); {len(signals)} remain.{removed_lines}\n", flush=True)

    log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance, vif, bic = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)
    model = get_model(t, d, signals)

    save_signal_diagnostics(path / "result", fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif)
    plot_time_series("Time Series", path / "result" / f"time_series_{len(signals)}.png", t, d, signals)
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "result" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals, core_edges=core_edges, filter_edges=filter_edges)
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "result" / f"fourier_space_{len(signals)}_model.png", t, model, f_min, f_max, core_edges=core_edges, filter_edges=filter_edges)
    plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "result" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc, Qs_unc, core_edges=core_edges, f_range=(f_min, f_max), filter_edges=filter_edges)
    plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "result" / f"covariance_{len(signals)}.png", signals, covariance)

    final_elapsed = perf_counter() - final_start
    posterior_variance = get_posterior_variance(t, d, B, signals)

    # Ownership: keep only signals whose frequency lies in this colony's core band.
    fs_all = np.asarray(signals)[:, 0]
    owned = (fs_all >= core_min) & ((fs_all <= core_max) if closed_above else (fs_all < core_max))
    core_signals = jnp.asarray(np.asarray(signals)[owned])
    core_bounds = jnp.asarray(np.asarray(bounds)[owned])

    print(
        f"{tag} Final stage completed in {final_elapsed:.2f} seconds\n"
        f"{tag} Colony completed in {perf_counter() - start_time:.2f} seconds: kept {int(owned.sum())} of {len(signals)} signal(s) in the core band\n",
        flush=True,
    )

    # Marginal record: this colony's own statistics (its data, filter, sigma and NUTS run) for the
    # signals it keeps. get_statistics returns them f-sorted, so ownership is re-applied to that order.
    fs_sorted = np.asarray(fs)
    owned_sorted = (fs_sorted >= core_min) & ((fs_sorted <= core_max) if closed_above else (fs_sorted < core_max))
    colony_summary.update(
        selected=best_count, kept=int(owned.sum()), glob_ll=float(glob_ll), bic=float(bic), noise_scale=posterior_variance,
    )
    marginal = make_marginal(colony_summary, owned_sorted, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, vif, covariance)

    return core_signals, core_bounds, posterior_variance, marginal

def run(type, event, network, station, f_min, f_max, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices, cores, diagnostics, signal_step, subbands, path, min_novelty=0.0, prune=False, fit_edge_gain=DEFAULT_FIT_EDGE_GAIN, ringing_db=DEFAULT_RINGING_DB, max_trim_hours=DEFAULT_MAX_TRIM_SECONDS / 3600, filter_order=DEFAULT_FILTER_ORDER, core_gain=DEFAULT_CORE_GAIN, max_condition=DEFAULT_MAX_CONDITION, model_error_db=DEFAULT_MODEL_ERROR_DB):
    start_time = perf_counter()
    t_raw, d_raw = load_data(type, event, network, station)
    max_trim_seconds = max_trim_hours * 3600
    # The full band is filtered like a colony: widened so the band edges keep >= core_gain of the amplitude.
    sampling_rate = 1 / np.mean(np.diff(t_raw))
    full_filter_min, full_filter_max, full_edge_gain = get_filter_band(f_min, f_max, sampling_rate, filter_order, core_gain)
    t, d = filter_data(t_raw, d_raw, full_filter_min, full_filter_max, order=filter_order, ringing_db=ringing_db, max_trim_seconds=max_trim_seconds)
    B = full_filter_max - full_filter_min
    print(
        f"Full band {1000 * f_min:.3f}-{1000 * f_max:.3f} mHz; filter {1000 * full_filter_min:.3f}-{1000 * full_filter_max:.3f} mHz "
        f"(order {filter_order}, {full_edge_gain:.3f} of amplitude kept at the band edges)\n",
        flush=True,
    )

    edges_init, band_edges = get_band_edges(path, t, d, f_min, f_max, Q_min, Q_max, 100, 100, subbands)
    plot_fourier_space("Fourier Space", path / "fourier_space.png", t, d, f_min, f_max, edges_init=edges_init, band_edges=band_edges)
    plot_filter_responses(path / "filter_response.png", band_edges, f_min, f_max, sampling_rate, filter_order, core_gain, fit_edge_gain)
    plot_time_series("Time Series", path / "time_series.png", t, d)

    numpyro.set_host_device_count(devices)

    signals, bounds, subband_variances, marginals = colony_orchestrator(path, t_raw, d_raw, band_edges, Q_min, Q_max, f_points, Q_points, f_margin, log_Q_margin, depth, warmup, sample, devices, cores, diagnostics, signal_step, min_novelty, prune, fit_edge_gain, ringing_db, max_trim_seconds, filter_order, core_gain, max_condition, model_error_db)

    signals = jnp.asarray(signals).reshape(-1, 2)

    # Colony boxes may reach into the padding past the global band; every kept signal lies in
    # a core band, so clipping the boxes to [f_min, f_max] keeps each signal inside its box.
    bounds = np.array(bounds, dtype=float).reshape(-1, 4)   # writable copy; np.asarray of a JAX array is read-only
    bounds[:, 0] = np.maximum(bounds[:, 0], f_min)
    bounds[:, 1] = np.minimum(bounds[:, 1], f_max)
    bounds = jnp.asarray(bounds)

    # Marginal result: each colony's own results for the signals it keeps, appended in subband order.
    # Nothing is refitted: every value comes from the colony that owns the signal (its filtered data,
    # sigma and NUTS run). Amplitudes and phases are moved from each colony's own clock to one common
    # reference time, the end of the longest trim. The covariance is block-diagonal: correlations
    # between subbands are not estimated.
    reference_hours = shift_marginals_to_reference(marginals)
    print(f"Marginal amplitudes and phases are at t = {reference_hours:.3f} h after the record start (the longest subband trim).\n", flush=True)
    marginal = append_marginals(marginals)
    marginal_signals = jnp.stack((jnp.asarray(marginal["fs"]), jnp.asarray(marginal["Qs"])), axis=-1).reshape(-1, 2)
    model = get_model(t, d, marginal_signals)   # display only: amplitudes refitted to the full-band data

    save_signal_diagnostics(path / "marginal_result", marginal["fs"], marginal["Qs"], marginal["fs_unc"], marginal["Qs_unc"], marginal["amplitudes"], marginal["phases"], marginal["vif"], extra_columns={"Subband": marginal["subband"], "Reference Time (h)": np.full(len(marginal["fs"]), reference_hours)})
    save_subband_summary(path / "marginal_result", [colony_marginal["summary"] for colony_marginal in marginals])
    plot_time_series("Time Series", path / "marginal_result" / f"time_series_{len(marginal_signals)}.png", t, d, marginal_signals)
    plot_fourier_space(f"Fourier Space {len(marginal_signals)} Signal Model", path / "marginal_result" / f"fourier_space_{len(marginal_signals)}.png", t, d, f_min, f_max, marginal_signals, core_edges=band_edges, edges_label="Subband boundaries")
    plot_fourier_space(f"Fourier Space {len(marginal_signals)} Signal Model", path / "marginal_result" / f"fourier_space_{len(marginal_signals)}_model.png", t, model, f_min, f_max, core_edges=band_edges, edges_label="Subband boundaries")
    plot_signal_space(f"f-Q Diagram {len(marginal_signals)} Signal Model", path / "marginal_result" / f"signal_space_{len(marginal_signals)}.png", marginal["fs"], marginal["Qs"], marginal["fs_unc"], marginal["Qs_unc"], core_edges=band_edges, f_range=(f_min, f_max), edges_label="Subband boundaries")
    plot_covariance(f"Parameter Correlation {len(marginal_signals)} Signal Model", path / "marginal_result" / f"covariance_{len(marginal_signals)}.png", marginal_signals, marginal["covariance"])
    print(f"Marginal result: {len(marginal_signals)} signals appended from {len(marginals)} subband(s)\n", flush=True)

    # Joint-run input checks; full arrays are printed only if a check fails.
    s = np.asarray(jax.device_get(signals))
    b = np.asarray(jax.device_get(bounds))
    lo = b[:, [0, 2]]  # f_lo, Q_lo
    hi = b[:, [1, 3]]  # f_hi, Q_hi
    finite = bool(np.isfinite(s).all() and np.isfinite(b).all())
    ordered = bool(np.all(lo < hi))
    within = (s >= lo) & (s <= hi)

    print(
        f"[joint] NUTS inputs: {len(s)} signals ({2 * len(s)} parameters), bounds shape={b.shape}\n"
        f"[joint]   finite={finite}, bounds ordered={ordered}, signals within bounds={bool(within.all())}\n",
        flush=True,
    )

    if not (finite and ordered and within.all()):
        print("signals:\n", s, "\nbounds:\n", b, "\nper-signal in-bounds:\n", within, "\n", flush=True)

    # The joint run always uses one chain, leaving it as much memory as possible.
    print(f"[joint] Starting NUTS at t={perf_counter() - start_time:.2f}s (warmup={warmup}, sample={sample}, chains=1)\n", flush=True)
    nuts_start = perf_counter()
    try:
        nuts_result, bounds, nuts_diagnostics = nuts(t, d, B, signals, bounds, warmup, sample, 1, return_diagnostics=True, tag="[joint]")

        # JAX may dispatch work asynchronously; wait before stopping the timer.
        nuts_result = jax.tree_util.tree_map(
            lambda x: x.block_until_ready()
            if hasattr(x, "block_until_ready") else x,
            nuts_result,
        )
        print(f"[joint] NUTS returned in {perf_counter() - nuts_start:.2f}s\n", flush=True)
    except Exception as error:
        out_of_memory = isinstance(error, MemoryError) or "RESOURCE_EXHAUSTED" in str(error) or "out of memory" in str(error).lower()

        if out_of_memory:
            print(
                f"[joint] NUTS ran out of memory after {perf_counter() - nuts_start:.2f}s with {2 * len(s)} parameters.\n"
                f"[joint] Marginal results are saved in {path / 'marginal_result'}; skipping the joint result.\n"
                f"Dracula completed (without joint result) in {perf_counter() - start_time:.2f} seconds",
                flush=True,
            )
            return

        print(f"[joint] NUTS failed after {perf_counter() - nuts_start:.2f}s\n", flush=True)
        traceback.print_exc()
        raise

    signals = nuts_result

    log_prob, glob_ll, noise_variance, snr, k2, fs, Qs, fs_unc, Qs_unc, amplitudes, phases, covariance, vif, bic = get_statistics(t, d, B, signals, f_min, f_max, Q_min, Q_max, max_condition=max_condition, model_error_db=model_error_db)
    model = get_model(t, d, signals)
    full_variance = get_posterior_variance(t, d, B, signals)
    covariance, signal_scale, subband_index = get_subband_scaled_covariance(signals, covariance, band_edges, subband_variances, full_variance)

    save_iteration_diagnostics(path / "joint_result", signals, noise_variance, snr, log_prob, glob_ll, bic, k2)
    save_signal_diagnostics(path / "joint_result", fs, Qs, fs_unc * signal_scale, Qs_unc * signal_scale, amplitudes, phases, vif)
    save_joint_uncertainties(path / "joint_result", fs, Qs, fs_unc, Qs_unc, signal_scale, subband_index)
    save_nuts_diagnostics(path / "joint_result", nuts_diagnostics, fs_unc, Qs_unc)
    print_nuts_diagnostics(nuts_diagnostics, tag="[joint]")
    plot_time_series("Time Series", path / "joint_result" / f"time_series_{len(signals)}.png", t, d, signals)
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "joint_result" / f"fourier_space_{len(signals)}.png", t, d, f_min, f_max, signals, core_edges=band_edges, edges_label="Subband boundaries")
    plot_fourier_space(f"Fourier Space {len(signals)} Signal Model", path / "joint_result" / f"fourier_space_{len(signals)}_model.png", t, model, f_min, f_max, core_edges=band_edges, edges_label="Subband boundaries")
    plot_signal_space(f"f-Q Diagram {len(signals)} Signal Model", path / "joint_result" / f"signal_space_{len(signals)}.png", fs, Qs, fs_unc * signal_scale, Qs_unc * signal_scale, core_edges=band_edges, f_range=(f_min, f_max), edges_label="Subband boundaries")
    plot_covariance(f"Parameter Correlation {len(signals)} Signal Model", path / "joint_result" / f"covariance_{len(signals)}.png", signals, covariance)

    print(f"Dracula completed in {perf_counter() - start_time:.2f} seconds")

    return

if __name__ == "__main__":
    config_file = Path("config.toml")

    with config_file.open("rb") as f:
        config = tomllib.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = (f"dracula_{config['type']}_{config['event']}_{config['network']}_{config['station']}_{timestamp}")
    path = Path.cwd() / name
    path.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(config_file, path / "config.txt")

    run(
        type=config["type"],
        event=config["event"],
        network=config["network"],
        station=config["station"],

        f_min=config["f_min"],
        f_max=config["f_max"],
        Q_min=config["Q_min"],
        Q_max=config["Q_max"],
        f_points=config["f_points"],
        Q_points=config["Q_points"],

        f_margin=config["f_margin"],
        log_Q_margin=config["log_Q_margin"],

        depth=config["depth"],
        warmup=config["warmup"],
        sample=config["sample"],

        devices=config["devices"],
        cores=config["cores"],

        diagnostics=config["diagnostics"],
        signal_step=config["signal_step"],
        subbands=config["subbands"],

        path=path,

        min_novelty=config.get("min_novelty", 0.1),
        prune=config.get("prune", True),
        fit_edge_gain=config.get("fit_edge_gain", DEFAULT_FIT_EDGE_GAIN),
        ringing_db=config.get("ringing_db", DEFAULT_RINGING_DB),
        max_trim_hours=config.get("max_trim_hours", DEFAULT_MAX_TRIM_SECONDS / 3600),
        filter_order=config.get("filter_order", DEFAULT_FILTER_ORDER),
        core_gain=config.get("core_gain", DEFAULT_CORE_GAIN),
        max_condition=config.get("max_condition", DEFAULT_MAX_CONDITION),
        model_error_db=config.get("model_error_db", DEFAULT_MODEL_ERROR_DB),
    )
