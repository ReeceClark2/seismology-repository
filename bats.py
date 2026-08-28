from __future__ import annotations

import inspect
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator, Mapping

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.special as jsp
from jax.scipy.special import logsumexp
from numpy.typing import ArrayLike

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def as_1d_float(value: ArrayLike, name: str) -> jax.Array:
    """Coerce ``value`` to a 1-D float64 JAX array."""
    array = jnp.ravel(jnp.asarray(value, dtype=jnp.float64))
    if array.ndim != 1:
        raise ValueError(f"{name} must be a scalar or 1-D array, got shape {array.shape}")
    return array


def prefix_bandwidth(
    value: float | ArrayLike | None,
    n: int,
    name: str,
) -> float | jax.Array | None:
    """Take the first ``n`` bandwidths, or pass a scalar through unchanged."""
    if value is None:
        return None
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim == 0:
        return array
    array = jnp.ravel(array)
    if array.shape[0] < n:
        raise ValueError(f"{name} has length {array.shape[0]}, expected at least {n}")
    return array[:n]


def broadcast_bandwidth(
    value: float | ArrayLike,
    n: int,
    name: str,
) -> jax.Array:
    """Broadcast a scalar bandwidth to length ``n``, or validate a 1-D array."""
    if n < 0:
        raise ValueError(f"{name}: expected a non-negative length, got {n}")
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim == 0:
        return jnp.full((n,), array)
    array = jnp.ravel(array)
    if array.shape[0] != n:
        raise ValueError(f"{name} has length {array.shape[0]}, expected {n}")
    return array


def _callable_named_kwargs(func: Any) -> set[str]:
    """Named keyword parameters of ``func``, ignoring ``**kwargs`` catch-alls."""
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return set()
    kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )
    return {
        name
        for name, param in parameters.items()
        if param.kind in kinds and name not in {"self", "cls"}
    }


def split_numpyro_kwargs(
    kwargs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Route NumPyro knobs into NUTS, MCMC, and ``MCMC.run`` dicts.

    Nested ``nuts_kwargs``, ``mcmc_kwargs``, and ``run_kwargs`` are passed
    through unfiltered so newly added NumPyro parameters keep working. Flat
    keys are assigned by inspecting the live NumPyro signatures.
    """
    remaining = dict(kwargs)
    nuts_kwargs = dict(remaining.pop("nuts_kwargs", None) or {})
    mcmc_kwargs = dict(remaining.pop("mcmc_kwargs", None) or {})
    run_kwargs = dict(remaining.pop("run_kwargs", None) or {})

    nuts_names = _callable_named_kwargs(NUTS) - {"model"}
    mcmc_names = _callable_named_kwargs(MCMC)
    run_names = _callable_named_kwargs(MCMC.run) - {"rng_key"}

    for key, value in remaining.items():
        in_nuts = key in nuts_names
        in_mcmc = key in mcmc_names
        in_run = key in run_names
        if in_nuts and not in_mcmc and not in_run:
            nuts_kwargs.setdefault(key, value)
        elif in_mcmc and not in_nuts and not in_run:
            mcmc_kwargs.setdefault(key, value)
        elif in_run and not in_nuts and not in_mcmc:
            run_kwargs.setdefault(key, value)
        elif in_nuts or in_mcmc or in_run:
            raise TypeError(
                f"Ambiguous NumPyro argument {key!r}; "
                "pass it in nuts_kwargs, mcmc_kwargs, or run_kwargs."
            )
        else:
            run_kwargs.setdefault(key, value)

    return nuts_kwargs, mcmc_kwargs, run_kwargs


def progress_bar_label(freq_lo: int, freq_hi: int, n_signals: int) -> str:
    return (
        f"Progress bar for frequencies {freq_lo}-{freq_hi} "
        f"for {n_signals} signal model"
    )


@contextmanager
def positioned_tqdm(desc: str, position: int | None) -> Iterator[None]:
    """Force NumPyro's tqdm bars onto a dedicated terminal row with a custom label."""
    if position is None:
        yield
        return

    import tqdm
    import tqdm.auto
    import tqdm.std

    tqdm.tqdm.monitor_interval = 0
    tqdm.auto.tqdm.monitor_interval = 0

    original_std = tqdm.std.tqdm
    original_auto = tqdm.auto.tqdm

    class PositionedTqdm(original_std):  # type: ignore[valid-type,misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["position"] = position
            kwargs.setdefault("leave", False)
            kwargs.setdefault("dynamic_ncols", True)
            kwargs.setdefault("mininterval", 0.2)
            kwargs.setdefault("file", sys.stderr)
            incoming = kwargs.get("desc") or ""
            if incoming and incoming not in desc:
                kwargs["desc"] = f"{desc}: {incoming}"
            else:
                kwargs["desc"] = desc
            super().__init__(*args, **kwargs)

        def set_description(self, desc_text: str | None = None, refresh: bool = True) -> None:
            if desc_text and desc not in str(desc_text):
                desc_text = f"{desc}: {desc_text}"
            elif not desc_text:
                desc_text = desc
            super().set_description(desc_text, refresh=refresh)

    replacements: list[tuple[Any, str, Any]] = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        try:
            current = getattr(module, "tqdm", None)
        except Exception:
            continue
        if current is original_std or current is original_auto:
            setattr(module, "tqdm", PositionedTqdm)
            replacements.append((module, "tqdm", current))

    try:
        yield
    finally:
        for module, name, previous in replacements:
            try:
                setattr(module, name, previous)
            except Exception:
                pass


@dataclass
class BATSResult:
    """Point estimate returned by a BATS NUTS run.

    Extra sampler output lives in ``extras`` so new fields can be added
    without breaking existing attribute access.
    """
    fs: jax.Array
    ks: jax.Array
    seed: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatisticsResult:
    """Global Bretthorst diagnostics for a fitted N-signal model."""
    log_prob: jax.Array
    variance: jax.Array
    SNR: jax.Array
    p_spec: jax.Array
    glob_LL: jax.Array
    fs: jax.Array
    ks: jax.Array
    f_unc: jax.Array
    k_unc: jax.Array
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        if key in self.extras:
            return self.extras[key]
        raise KeyError(key)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@jax.jit
def get_log_prob(t: jax.Array, d: jax.Array, fs: jax.Array, ks: jax.Array) -> jax.Array:
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
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Calculate the ratio of the squared projections to the squared data.
    # Note: The m and N terms cancel out here compared to calculating the 
    # explicit mean square data (msd) and mean square projection (msp).
    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    ratio = sum_sq_proj / sum_sq_data
    
    return 0.5 * (m - N) * jnp.log(1.0 - ratio)


@jax.jit
def get_model(t: jax.Array, d: jax.Array, fs: jax.Array, ks: jax.Array) -> jax.Array:
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
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    model = jnp.zeros(len(t))
    for ind, _ in enumerate(h):
        model += h[ind] * H[ind]

    return model


def bats_model(
    t: jax.Array,
    d: jax.Array,
    f_loc: jax.Array,
    f_scale: jax.Array,
    k_loc: jax.Array,
    k_scale: jax.Array,
    prior_n_std: float = 5.0,
) -> None:
    n_std = jnp.asarray(prior_n_std, dtype=f_loc.dtype)
    f_low = f_loc - n_std * f_scale
    f_high = jnp.maximum(f_loc + n_std * f_scale, f_low + 1e-12)
    k_low = jnp.maximum(0.0, k_loc - n_std * k_scale)
    k_high = jnp.maximum(k_loc + n_std * k_scale, k_low + 1e-12)

    fs = numpyro.sample(
        "fs",
        dist.TruncatedNormal(f_loc, f_scale, low=f_low, high=f_high).to_event(1),
    )
    ks = numpyro.sample(
        "ks",
        dist.TruncatedNormal(k_loc, k_scale, low=k_low, high=k_high).to_event(1),
    )

    numpyro.factor("bretthorst", get_log_prob(t, d, fs, ks))


def get_statistics(
    t: ArrayLike,
    d: ArrayLike,
    fs: ArrayLike,
    ks: ArrayLike,
) -> StatisticsResult:
    t = as_1d_float(t, "t")
    d = as_1d_float(d, "d")
    fs = as_1d_float(fs, "fs")
    ks = as_1d_float(ks, "ks")

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
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Calculate the ratio of the squared projections to the squared data.
    # Note: The m and N terms cancel out here compared to calculating the 
    # explicit mean square data (msd) and mean square projection (msp).
    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)

    ratio = (m / N) * mean_sq_proj / mean_sq_data

    log_prob = 0.5 * (m - N) * jnp.log(1.0 - ratio)

    # Variance --------------------------------------------------
    
    variance = (1 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2))

    # SNR -------------------------------------------------------
    
    SNR = ((m / N) * (1 + mean_sq_proj / variance)) ** (0.5)

    # Uncertainties (Bretthorst Eq. 4.11 and 4.13) -------------

    def log_prob_wrapper(q):
        return get_log_prob(t, d, q[:r], q[r:])

    q = jnp.concatenate((jnp.asarray(fs), jnp.asarray(ks)))
    log_prob_hessian = jax.jit(jax.hessian(log_prob_wrapper))(q)
    b_unc = (-m / 2.0) * log_prob_hessian
    evals_unc, evecs_unc = jnp.linalg.eigh(b_unc)
    evals_unc = jnp.maximum(evals_unc, 1e-8)
    param_unc = jnp.sqrt(
        jnp.maximum(variance, 0.0) * jnp.sum((evecs_unc ** 2) / evals_unc, axis=1)
    )
    f_unc = param_unc[:r]
    k_unc = param_unc[r:]

    # Power Spectrum ---------------------------------------------

    f_space = jnp.append(jnp.linspace(0.98 * min(fs), 1.02 * max(fs), 10_000), fs)

    def compute_C_single(f_val):
        phase = 2 * jnp.pi * f_val * t
        return (1 / N) * jnp.abs(jnp.sum(d * jnp.exp(1j * phase))) ** 2

    C = jax.lax.map(compute_C_single, f_space)
    
    def ms_projection_wrapper(q):
        f = q[:r]
        a = q[r:]  # decay rates; assumed positive

        omega = 2.0 * jnp.pi * f

        arg = omega[:, None] * t[None, :]
        decay = jnp.exp(-a[:, None] * t[None, :])

        G = jnp.vstack((
            jnp.cos(arg) * decay,
            jnp.sin(arg) * decay
        ))

        # G G^T
        M = G @ G.T

        # Data projection
        proj_d = G @ d

        # Scale-dependent ridge for numerical stability
        ridge = 1e-8 * jnp.trace(M) / M.shape[0]
        M_reg = M + ridge * jnp.eye(M.shape[0])

        # Solve M_reg x = G d
        x = jnp.linalg.solve(M_reg, proj_d)

        # Projection power
        return jnp.dot(proj_d, x) / m

    hessian = jax.jit(
        jax.hessian(ms_projection_wrapper)
    )(q)

    hessian_diag = jnp.diag(hessian)[:r]
    b_diagonal = (-m / 2.0) * hessian_diag

    # p_space = (2 * (variance + jnp.sum(C)) * 
    #           jnp.sum((b_diagonal[:, None] / (2 * jnp.pi * variance)) ** (1 / 2) * 
    #           jnp.exp((-b_diagonal[:, None] * (fs[:, None] - f_space) ** 2) / (2 * variance)), axis=0))

    # 1. SAFEGUARDS: Prevent log(negative) and log(0)
    # Force b_diagonal to be strictly positive and > 0
    safe_b_diag = jnp.maximum(jnp.abs(b_diagonal), 1e-30)

    # Force variance to be > 0 to prevent division by zero or log(0)
    safe_var = jnp.maximum(variance, 1e-30)

    # 2. Compute the log of the amplitude factor using safe variables
    log_amplitude = 0.5 * (jnp.log(safe_b_diag[:, None]) - jnp.log(2 * jnp.pi * safe_var))

    # 3. The exponent uses the safe variables
    exponent = (-safe_b_diag[:, None] * (fs[:, None] - f_space) ** 2) / (2 * safe_var)

    # 4. Combine them in log space
    X = log_amplitude + exponent

    # 5. Use the LSE trick directly
    log_inner_sum = logsumexp(X, axis=0)

    # 6. Compute the log of the leading constant scalar
    # Also safeguard the sum of C just in case it dipped negative
    safe_C_sum = jnp.maximum(jnp.sum(jnp.nan_to_num(C)), 0.0)
    log_constant = jnp.log(2 * (safe_var + safe_C_sum))

    # 7. Add them together to get the final log-spectrum
    log_p_space = log_constant + log_inner_sum

    # 8. Convert back to linear space
    p_space = jnp.exp(log_p_space)
    p_spec = jnp.column_stack((f_space, p_space))

    # Global Likelihood -----------------------------------------

    R_delta = float(jnp.max(jnp.abs(d)))
    R_sigma = float(jnp.max(jnp.abs(d)))

    log_R_delta = jnp.maximum(jnp.log(R_delta), 1e-12)
    log_R_sigma = jnp.maximum(jnp.log(R_sigma), 1e-12)

    R_gamma = (0.5 / float(jnp.mean(jnp.diff(t)))) * float(t[-1] - t[0])

    b = (-m / 2.0) * hessian
    eigenvalues, _ = jnp.linalg.eigh(b)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    factor = ((m / 2.0) * jnp.log(2.0 * jnp.pi)
                        - 0.5 * jnp.sum(jnp.log(eigenvalues))
                        - m * jnp.log(R_gamma))
    delta_term = (jsp.gammaln(m / 2.0)
                    - jnp.log(2.0 * log_R_delta)
                    - (m / 2.0) * jnp.log((m * mean_sq_proj) / 2.0))
    sigma_term = (jsp.gammaln((N - m - r) / 2.0)
                    - jnp.log(2.0 * log_R_sigma)
                    - ((N - m - r) / 2.0) * jnp.log((N * mean_sq_data - m * mean_sq_proj) / 2.0))
    gamma_term = -(2.0 * r) * jnp.log(R_gamma)

    glob_LL = delta_term + sigma_term + gamma_term + factor

    return StatisticsResult(
        log_prob=log_prob,
        variance=variance,
        SNR=SNR,
        p_spec=p_spec,
        glob_LL=glob_LL,
        fs=fs,
        ks=ks,
        f_unc=f_unc,
        k_unc=k_unc,
    )


def rank_by_power(
    t: ArrayLike,
    d: ArrayLike,
    fs: ArrayLike,
    ks: ArrayLike,
) -> tuple[jax.Array, jax.Array, StatisticsResult]:
    """Return strongest-to-weakest indices from the Bretthorst power spectrum."""
    fs = as_1d_float(fs, "fs")
    ks = as_1d_float(ks, "ks")
    stats = get_statistics(t, d, fs, ks)
    f_grid, p_grid = stats.p_spec.T
    p_idx = jax.vmap(lambda target: jnp.argmin(jnp.abs(f_grid - target)))(fs)
    powers = p_grid[p_idx]
    order = jnp.argsort(powers)[::-1]
    return order, powers, stats


class BATS:
    def __init__(
        self,
        t: ArrayLike,
        d: ArrayLike,
        f_init: ArrayLike,
        k_init: ArrayLike,
    ) -> None:
        self.t = as_1d_float(t, "t")
        self.d = as_1d_float(d, "d")
        if self.t.shape[0] != self.d.shape[0]:
            raise ValueError(
                f"t and d must have the same length, got {self.t.shape[0]} and {self.d.shape[0]}"
            )

        self.f_init = as_1d_float(f_init, "f_init")
        self.k_init = as_1d_float(k_init, "k_init")
        if self.f_init.shape[0] != self.k_init.shape[0]:
            raise ValueError(
                "f_init and k_init must have the same length, "
                f"got {self.f_init.shape[0]} and {self.k_init.shape[0]}"
            )

    def run_NUTS(
        self,
        f_bw: float | ArrayLike,
        k_bw: float | ArrayLike,
        W: int = 1_000,
        S: int = 2_000,
        seed: int = 0,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        prior_n_std: float = 5.0,
        **kwargs: Any,
    ) -> BATSResult:
        n = int(self.f_init.shape[0])
        f_bw = broadcast_bandwidth(f_bw, n, "f_bw")
        k_bw = broadcast_bandwidth(k_bw, n, "k_bw")
        if prior_n_std <= 0:
            raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")

        nuts_kwargs, mcmc_kwargs, run_kwargs = split_numpyro_kwargs(kwargs)

        init_strategy = init_to_value(values={"fs": self.f_init, "ks": self.k_init})

        nuts_config: dict[str, Any] = {
            "init_strategy": init_strategy,
            "dense_mass": True,
            "target_accept_prob": 0.8,
            "max_tree_depth": 8,
        }
        nuts_config.update(nuts_kwargs)
        nuts_config.pop("model", None)

        mcmc_config: dict[str, Any] = {
            "num_warmup": W,
            "num_samples": S,
            "num_chains": 1,
            "progress_bar": True,
        }
        mcmc_config.update(mcmc_kwargs)

        extra_fields = run_kwargs.pop("extra_fields", ("potential_energy",))
        if "potential_energy" not in extra_fields:
            extra_fields = tuple(extra_fields) + ("potential_energy",)

        if progress_desc is None:
            progress_desc = progress_bar_label(1, n, n)

        kernel = NUTS(bats_model, **nuts_config)

        mcmc = MCMC(kernel, **mcmc_config)

        show_bar = bool(mcmc_config.get("progress_bar", True))
        bar_position = progress_position if show_bar else None

        with positioned_tqdm(progress_desc, bar_position):
            mcmc.run(
                jax.random.PRNGKey(int(seed)),
                self.t,
                self.d,
                self.f_init,
                f_bw,
                self.k_init,
                k_bw,
                prior_n_std,
                extra_fields=extra_fields,
                **run_kwargs,
            )

        samples = mcmc.get_samples()

        extra_fields_out = mcmc.get_extra_fields()
        pe = extra_fields_out["potential_energy"]
        best_ind = jnp.argmin(pe)

        best_fs = samples["fs"][best_ind]
        best_ks = samples["ks"][best_ind]

        extras = {
            "potential_energy": pe,
            "best_index": best_ind,
        }

        return BATSResult(fs=best_fs, ks=best_ks, seed=int(seed), extras=extras)
