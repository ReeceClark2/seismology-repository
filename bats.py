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

from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
from numpy.typing import ArrayLike
from typing import Any, Iterator, Literal, Mapping
from scipy.signal import butter, sosfilt

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints, transforms
from numpyro.infer import MCMC, NUTS, init_to_value

import tqdm
import tqdm.auto
import tqdm.std

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def _single_signal_model(
    t: ArrayLike,
    d: ArrayLike,
    frequency: float | jax.Array,
    decay_rate: float | jax.Array,
) -> jax.Array:
    """Return the least-squares damped sinusoid for one frequency and decay rate."""
    t_arr = jnp.asarray(t, dtype=jnp.float64)
    d_arr = jnp.asarray(d, dtype=jnp.float64)

    f = jnp.asarray(frequency, dtype=jnp.float64)
    k = jnp.asarray(decay_rate, dtype=jnp.float64)

    phase = 2.0 * jnp.pi * f * t_arr
    envelope = jnp.exp(-k * t_arr)

    cosine = jnp.cos(phase) * envelope
    sine = jnp.sin(phase) * envelope

    # Design matrix with shape (n_samples, 2).
    design = jnp.column_stack((cosine, sine))

    # Least-squares amplitudes for cosine and sine components.
    coefficients, _, _, _ = jnp.linalg.lstsq(
        design,
        d_arr,
        rcond=None,
    )

    return design @ coefficients


def _create_diagnostics_directory() -> Path:
    """Create a unique timestamped diagnostics directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = Path.cwd() / f"diagnostics_{timestamp}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _plot_grid_probability_diagnostic(
    path: Path,
    f_space: ArrayLike,
    k_space: ArrayLike,
    probabilities: ArrayLike,
    selected_f: float,
    selected_k: float,
    selected_probability: float,
    signal_number: int,
    selection: str,
) -> None:
    """Save the frequency/decay-rate probability surface as a heatmap."""
    f_np = np.asarray(f_space, dtype=float).ravel()
    k_np = np.asarray(k_space, dtype=float).ravel()
    probability_np = np.asarray(probabilities, dtype=float).reshape(
        f_np.size,
        k_np.size,
    )

    # Mask NaN and infinite values so one invalid point does not prevent
    # the rest of the probability surface from being plotted.
    probability_plot = np.ma.masked_invalid(probability_np)

    fig, ax = plt.subplots(figsize=(10, 7))

    mesh = ax.pcolormesh(
        k_np,
        f_np,
        probability_plot,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )

    ax.scatter(
        [selected_k],
        [selected_f],
        marker="x",
        s=100,
        linewidths=2.0,
        color="red",
        label=(
            f"Selected\n"
            f"f={selected_f:.6g} Hz\n"
            f"k={selected_k:.6g}"
        ),
    )

    colorbar = fig.colorbar(mesh, ax=ax)
    colorbar.set_label("Log Probability")

    ax.set_xlabel("Decay rate")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f"Grid Search Probability for Signal {signal_number}\n"
        f"Selected Log Probability = {selected_probability:.7g}"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def _hann_fft(
    t: ArrayLike,
    data: ArrayLike,
    min_f: float,
    max_f: float,
    target_points: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a zero-padded, Hann-windowed FFT restricted to a frequency band.

    ``target_points`` is the approximate number of frequency samples desired
    between ``min_f`` and ``max_f``. Zero-padding increases the density of the
    plotted FFT grid, but does not increase the underlying physical frequency
    resolution beyond the observation duration.
    """
    t_np = np.asarray(t, dtype=float).ravel()
    data_np = np.asarray(data, dtype=float).ravel()

    if t_np.size != data_np.size:
        raise ValueError(
            "t and data must have the same length for FFT diagnostics"
        )

    if t_np.size < 2:
        raise ValueError(
            "At least two samples are required for FFT diagnostics"
        )

    if min_f < 0 or max_f <= min_f:
        raise ValueError(
            f"Expected 0 <= min_f < max_f, got {min_f} and {max_f}"
        )

    if target_points < 2:
        raise ValueError(
            f"target_points must be >= 2, got {target_points}"
        )

    dt = np.diff(t_np)

    if np.any(dt <= 0):
        raise ValueError(
            "t must be strictly increasing for FFT diagnostics"
        )

    median_dt = float(np.median(dt))
    sampling_rate = 1.0 / median_dt
    nyquist = 0.5 * sampling_rate

    if max_f > nyquist:
        raise ValueError(
            f"max_f ({max_f}) exceeds the Nyquist frequency ({nyquist})"
        )

    if not np.allclose(
        dt,
        median_dt,
        rtol=1e-3,
        atol=max(1e-12, abs(median_dt) * 1e-6),
    ):
        warnings.warn(
            "The time samples are not uniformly spaced. FFT diagnostic "
            "frequencies are being calculated using the median time step.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Choose an FFT length that gives approximately target_points samples
    # across the requested frequency range.
    desired_df = (max_f - min_f) / float(target_points - 1)
    required_fft_length = int(np.ceil(sampling_rate / desired_df))

    # Use a power of two for efficient FFT execution.
    fft_length = max(t_np.size, required_fft_length)
    fft_length = 1 << (fft_length - 1).bit_length()

    window = np.hanning(t_np.size)
    window_sum = float(np.sum(window))

    if window_sum <= 0:
        raise ValueError(
            "Unable to normalize the Hann window for FFT diagnostics"
        )

    transformed = np.fft.rfft(
        data_np * window,
        n=fft_length,
    )

    frequencies = np.fft.rfftfreq(
        fft_length,
        d=median_dt,
    )

    # One-sided amplitude normalization.
    amplitude = 2.0 * np.abs(transformed) / window_sum
    amplitude[0] *= 0.5

    if fft_length % 2 == 0 and amplitude.size > 1:
        amplitude[-1] *= 0.5

    # Explicitly restrict the returned FFT to [min_f, max_f].
    mask = (
        (frequencies >= min_f)
        & (frequencies <= max_f)
    )

    return frequencies[mask], amplitude[mask]


def _plot_timeseries_diagnostic(
    path: Path,
    t: ArrayLike,
    data_before: ArrayLike,
    data_after: ArrayLike,
    model: ArrayLike,
    selected_f: float,
    selected_k: float,
    signal_number: int,
) -> None:
    """Save time-domain data before and after model subtraction."""
    t_np = np.asarray(t, dtype=float).ravel()
    before_np = np.asarray(data_before, dtype=float).ravel()
    after_np = np.asarray(data_after, dtype=float).ravel()
    model_np = np.asarray(model, dtype=float).ravel()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    axes[0].plot(
        t_np,
        before_np,
        color="black",
        linewidth=0.8,
        label="Before Model Subtraction",
    )
    axes[0].plot(
        t_np,
        model_np,
        color="red",
        linewidth=0.3,
        label="Selected Model",
    )
    axes[0].set_ylabel("Intensity")
    axes[0].set_title(
        f"Signal {signal_number} Removal Time Domain\n"
        f"f={selected_f:.6g} Hz, k={selected_k:.6g}"
    )
    axes[0].legend(loc="best")
    # axes[0].grid(alpha=0.2)

    axes[1].plot(
        t_np,
        after_np,
        color="C0",
        linewidth=0.8,
        label="After Model Subtraction",
    )
    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Residual")
    axes[1].legend(loc="best")
    # axes[1].grid(alpha=0.2)

    fig.tight_layout()

    try:
        fig.savefig(path, dpi=175, bbox_inches="tight")
    finally:
        plt.close(fig)


def _plot_fourier_diagnostic(
    path: Path,
    t: ArrayLike,
    data_before: ArrayLike,
    data_after: ArrayLike,
    selected_f: float,
    min_f: float,
    max_f: float,
    signal_number: int,
    fft_points: int = 8192,
) -> None:
    """Save Hann-windowed FFTs before and after model subtraction."""
    frequencies_before, amplitude_before = _hann_fft(
        t,
        data_before,
        min_f=min_f,
        max_f=max_f,
        target_points=fft_points,
    )

    frequencies_after, amplitude_after = _hann_fft(
        t,
        data_after,
        min_f=min_f,
        max_f=max_f,
        target_points=fft_points,
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        frequencies_before,
        amplitude_before,
        color="0.35",
        linewidth=1.0,
        label="Before Model Subtraction",
    )

    ax.plot(
        frequencies_after,
        amplitude_after,
        color="C0",
        linewidth=1.0,
        label="After Model Subtraction",
    )

    ax.axvline(
        selected_f,
        color="C3",
        linestyle="--",
        linewidth=1.2,
        label=f"Selected frequency: {selected_f:.7g} Hz",
    )

    ax.set_xlim(min_f, max_f)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(
        f"Signal {signal_number} Removal Frequency Domain"
    )
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=175, bbox_inches="tight")
    plt.close(fig)


def _butter_bandpass_filter(
    t: jax.Array,
    d: jax.Array,
    min_f: float,
    max_f: float,
    order: int,
) -> jax.Array:
    """Apply a causal Butterworth bandpass to data sampled at times ``t``."""
    t_np = np.asarray(t, dtype=float)
    d_np = np.asarray(d, dtype=float)

    if t_np.size < 2:
        raise ValueError("t must contain at least two samples for bandpass filtering")

    dt = np.diff(t_np)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing")

    sampling_rate = float(1.0 / np.median(dt))
    nyquist = 0.5 * sampling_rate

    if min_f <= 0:
        raise ValueError(
            f"min_f must be > 0 when applying a bandpass, got {min_f}"
        )
    if max_f >= nyquist:
        raise ValueError(
            f"max_f ({max_f}) must be below the Nyquist frequency ({nyquist})"
        )

    sos = butter(
        order,
        [min_f, max_f],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    filtered = sosfilt(sos, d_np)
    return jnp.asarray(filtered, dtype=jnp.float64)


def _single_signal_log_prob_batch(
    t: jax.Array,
    d: jax.Array,
    frequencies: jax.Array,
    decay_rates: jax.Array,
) -> jax.Array:
    """Evaluate one-signal Bretthorst probabilities for a batch of grid points.

    This is algebraically equivalent to calling ``get_log_prob`` separately
    for each frequency/decay-rate pair, but performs the operations in batch.
    """
    n_data = d.shape[0]
    m = 2

    phase = (
        2.0
        * jnp.pi
        * frequencies[:, None]
        * t[None, :]
    )
    decay = jnp.exp(-decay_rates[:, None] * t[None, :])

    cosine = jnp.cos(phase) * decay
    sine = jnp.sin(phase) * decay

    # Batched 2x2 Gram matrices for G = [cosine; sine].
    cc = jnp.sum(cosine * cosine, axis=1)
    cs = jnp.sum(cosine * sine, axis=1)
    ss = jnp.sum(sine * sine, axis=1)

    gram = jnp.stack(
        (
            jnp.stack((cc, cs), axis=-1),
            jnp.stack((cs, ss), axis=-1),
        ),
        axis=-2,
    )

    # G @ d for each grid point.
    projections = jnp.stack(
        (
            cosine @ d,
            sine @ d,
        ),
        axis=-1,
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Equivalent to computing h = H @ d without materializing H.
    rotated_projections = jnp.einsum(
        "bij,bj->bi",
        jnp.swapaxes(eigenvectors, -1, -2),
        projections,
    )
    sum_sq_proj = jnp.sum(
        rotated_projections**2 / eigenvalues,
        axis=-1,
    )

    sum_sq_data = jnp.maximum(jnp.sum(d**2), 1e-30)
    ratio = jnp.clip(
        sum_sq_proj / sum_sq_data,
        0.0,
        1.0 - 1e-12,
    )

    return 0.5 * (m - n_data) * jnp.log1p(-ratio)


@jax.jit
def _evaluate_single_signal_grid(
    t: jax.Array,
    d: jax.Array,
    frequency_batches: jax.Array,
    decay_batches: jax.Array,
) -> jax.Array:
    """Evaluate padded grid batches sequentially on-device.

    Each row contains one vectorized batch of frequency/decay-rate pairs.
    ``jax.lax.map`` avoids allocating an array proportional to the complete
    grid size times the number of time samples.
    """
    def evaluate_batch(inputs: tuple[jax.Array, jax.Array]) -> jax.Array:
        frequencies, decay_rates = inputs
        return _single_signal_log_prob_batch(
            t,
            d,
            frequencies,
            decay_rates,
        )

    return jax.lax.map(
        evaluate_batch,
        (frequency_batches, decay_batches),
    )


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
    cov_mat: jax.Array
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

    ratio = sum_sq_proj / jnp.maximum(sum_sq_data, 1e-30)
    ratio = jnp.clip(ratio, 0.0, 1.0 - 1e-12)

    return 0.5 * (m - N) * jnp.log1p(-ratio)


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
    eigenvalues = jnp.maximum(eigenvalues, 1e-19)

    # Bretthorst Eq. 3.6: orthonormal functions H
    T = (eigenvectors / jnp.sqrt(eigenvalues)).T
    
    H = T @ G
    h = H @ d
    model = h @ H

    # Transform orthogonal amplitudes (h) back to physical amplitudes (A)
    B = h @ T

    return B, model


def bats_model(
    t: jax.Array,
    d: jax.Array,
    f_loc: jax.Array,
    f_scale: jax.Array,
    k_loc: jax.Array,
    k_scale: jax.Array,
    prior_n_std: float = 5.0,
    unbounded: bool = False,
) -> None:

    n_std = jnp.asarray(prior_n_std, dtype=f_loc.dtype)
    
    f_low = f_loc - n_std * f_scale
    f_high = jnp.maximum(f_loc + n_std * f_scale, f_low + 1e-12)

    # k_loc must be positive for log-space sampling.
    k_floor = jnp.asarray(1e-12, dtype=k_loc.dtype)
    k_loc_safe = jnp.maximum(k_loc, k_floor)

    log_k_loc = jnp.log(k_loc_safe)
    log_k_low = log_k_loc - n_std * k_scale
    log_k_high = log_k_loc + n_std * k_scale

    if unbounded:
        fs = numpyro.sample(
            "fs",
            dist.Uniform(f_low, f_high).to_event(1),
        )

        ks = numpyro.sample(
            "ks",
            dist.TransformedDistribution(
                dist.Uniform(
                    log_k_low,
                    log_k_high,
                ).to_event(1),
                transforms.ExpTransform(),
            ),
        )
    else:
        fs = numpyro.sample(
            "fs",
            dist.TruncatedNormal(
                f_loc,
                f_scale,
                low=f_low,
                high=f_high,
            ).to_event(1),
        )

        ks = numpyro.sample(
            "ks",
            dist.TransformedDistribution(
                dist.TruncatedNormal(
                    log_k_loc,
                    k_scale,
                    low=log_k_low,
                    high=log_k_high,
                ).to_event(1),
                transforms.ExpTransform(),
            ),
        )

    numpyro.factor("bretthorst", get_log_prob(t, d, fs, ks))


def get_statistics(
    t: ArrayLike,
    d: ArrayLike,
    fs: ArrayLike,
    ks: ArrayLike,
    *,
    calc_log_prob: bool = True,
    calc_variance: bool = True,
    calc_snr: bool = True,
    calc_p_spec: bool = True,
    calc_glob_ll: bool = True,
    calc_cov_mat: bool = True,
    calc_f_unc: bool = True,
    calc_k_unc: bool = True,
) -> StatisticsResult:
    """Calculate selected Bretthorst statistics.

    Disabled results are returned as ``None``. Intermediate quantities may
    still be calculated when they are dependencies of an enabled result.

    For example, SNR requires the residual variance internally. Setting
    ``calc_variance=False`` prevents the variance from being returned, but
    it will still be calculated if ``calc_snr=True``.
    """
    t = as_1d_float(t, "t")
    d = as_1d_float(d, "d")
    fs = as_1d_float(fs, "fs")
    ks = as_1d_float(ks, "ks")

    if t.shape[0] != d.shape[0]:
        raise ValueError(
            "t and d must have the same length, "
            f"got {t.shape[0]} and {d.shape[0]}"
        )

    if fs.shape[0] != ks.shape[0]:
        raise ValueError(
            "fs and ks must have the same length, "
            f"got {fs.shape[0]} and {ks.shape[0]}"
        )

    r = int(fs.shape[0])
    m = 2 * r
    N = int(d.shape[0])

    if r < 1:
        raise ValueError("At least one frequency and decay rate are required")

    # Output values remain None unless their corresponding flag is enabled.
    log_prob_out: jax.Array | None = None
    variance_out: jax.Array | None = None
    snr_out: jax.Array | None = None
    p_spec_out: jax.Array | None = None
    glob_ll_out: jax.Array | None = None
    cov_mat_out: jax.Array | None = None
    f_unc_out: jax.Array | None = None
    k_unc_out: jax.Array | None = None

    # Internal values may be needed as dependencies without being returned.
    variance_internal: jax.Array | None = None
    mean_sq_data: jax.Array | None = None
    mean_sq_proj: jax.Array | None = None

    needs_variance = any(
        (
            calc_variance,
            calc_snr,
            calc_p_spec,
            calc_f_unc,
            calc_k_unc,
        )
    )

    needs_projection = any(
        (
            calc_log_prob,
            needs_variance,
            calc_glob_ll,
        )
    )

    # ------------------------------------------------------------------
    # Shared signal projection
    # ------------------------------------------------------------------
    if needs_projection:
        omegas = fs * 2.0 * jnp.pi

        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-ks[:, None] * t[None, :])

        # Non-orthogonal model matrix.
        G = jnp.vstack(
            (
                jnp.cos(arg) * decay,
                jnp.sin(arg) * decay,
            )
        )

        # Compress the time dimension before decomposing.
        _, R = jnp.linalg.qr(G.T, mode="reduced")
        U, singular_values, _ = jnp.linalg.svd(
            R.T,
            full_matrices=False,
        )

        eigenvalues = jnp.maximum(
            singular_values**2,
            1e-30,
        )
        eigenvectors = U

        # Bretthorst Eq. 3.6 and Eq. 3.13.
        H = (
            eigenvectors / jnp.sqrt(eigenvalues)
        ).T @ G
        h = H @ d

        sum_sq_data = jnp.sum(d**2)
        sum_sq_proj = jnp.sum(h**2)

        mean_sq_data = sum_sq_data / N
        mean_sq_proj = sum_sq_proj / m

        projection_ratio = (
            (m / N) * mean_sq_proj / mean_sq_data
        )

        if calc_log_prob:
            log_prob_out = (
                0.5
                * (m - N)
                * jnp.log(1.0 - projection_ratio)
            )

        if needs_variance:
            variance_internal = (
                sum_sq_data - sum_sq_proj
            ) / (N - m - 2)

            if calc_variance:
                variance_out = variance_internal

        if calc_snr:
            assert variance_internal is not None

            snr_out = jnp.sqrt(
                (m / N)
                * (
                    1.0
                    + mean_sq_proj / variance_internal
                )
            )

    # ------------------------------------------------------------------
    # Parameter Hessian, uncertainties, and covariance matrix
    # ------------------------------------------------------------------
    needs_parameter_hessian = any(
        (
            calc_f_unc,
            calc_k_unc,
            calc_cov_mat,
        )
    )

    if needs_parameter_hessian:

        def log_prob_wrapper(q: jax.Array) -> jax.Array:
            return get_log_prob(
                t,
                d,
                q[:r],
                q[r:],
            )

        q = jnp.concatenate((fs, ks))

        log_prob_hessian = jax.jit(
            jax.hessian(log_prob_wrapper)
        )(q)

        b_unc = (-m / 2.0) * log_prob_hessian

        evals_unc, evecs_unc = jnp.linalg.eigh(b_unc)
        evals_unc = jnp.maximum(evals_unc, 1e-8)

        if calc_f_unc or calc_k_unc:
            assert variance_internal is not None

            param_unc = jnp.sqrt(
                jnp.maximum(variance_internal, 0.0)
                * jnp.sum(
                    (evecs_unc**2) / evals_unc,
                    axis=1,
                )
            )

            if calc_f_unc:
                f_unc_out = param_unc[:r]

            if calc_k_unc:
                k_unc_out = param_unc[r:]

        if calc_cov_mat:
            inv_b_unc = (
                evecs_unc / evals_unc
            ) @ evecs_unc.T

            cov_mat_out = (m / 2.0) * inv_b_unc

    # ------------------------------------------------------------------
    # Mean-square projection Hessian shared by the power spectrum and
    # global likelihood.
    # ------------------------------------------------------------------
    projection_hessian: jax.Array | None = None

    if calc_p_spec or calc_glob_ll:
        q = jnp.concatenate((fs, ks))

        @jax.remat
        def ms_projection_wrapper(
            parameters: jax.Array,
        ) -> jax.Array:
            frequencies = parameters[:r]
            decay_rates = parameters[r:]

            omega = 2.0 * jnp.pi * frequencies
            arg = omega[:, None] * t[None, :]
            decay = jnp.exp(
                -decay_rates[:, None] * t[None, :]
            )

            G = jnp.vstack(
                (
                    jnp.cos(arg) * decay,
                    jnp.sin(arg) * decay,
                )
            )

            proj_d = G @ d
            gram = G @ G.T

            eigenvalues, eigenvectors = jnp.linalg.eigh(
                gram
            )

            ridge = (
                1e-8
                * jnp.sum(eigenvalues)
                / gram.shape[0]
            )

            y = eigenvectors.T @ proj_d
            z = y / (eigenvalues + ridge)
            coefficients = eigenvectors @ z

            return jnp.dot(
                proj_d,
                coefficients,
            ) / m

        projection_hessian = jax.jit(
            jax.hessian(ms_projection_wrapper)
        )(q)

    # ------------------------------------------------------------------
    # Power spectrum
    # ------------------------------------------------------------------
    if calc_p_spec:
        assert variance_internal is not None
        assert projection_hessian is not None

        f_space = jnp.append(
            jnp.linspace(
                0.98 * jnp.min(fs),
                1.02 * jnp.max(fs),
                10_000,
            ),
            fs,
        )

        def compute_C_single(
            frequency: jax.Array,
        ) -> jax.Array:
            phase = 2.0 * jnp.pi * frequency * t

            return (
                jnp.abs(
                    jnp.sum(
                        d * jnp.exp(1j * phase)
                    )
                )
                ** 2
                / N
            )

        C = jax.lax.map(
            compute_C_single,
            f_space,
        )

        hessian_diag = jnp.diag(
            projection_hessian
        )[:r]

        b_diagonal = (
            -m / 2.0
        ) * hessian_diag

        safe_b_diag = jnp.maximum(
            jnp.abs(b_diagonal),
            1e-30,
        )
        safe_variance = jnp.maximum(
            variance_internal,
            1e-30,
        )

        log_amplitude = 0.5 * (
            jnp.log(safe_b_diag[:, None])
            - jnp.log(
                2.0
                * jnp.pi
                * safe_variance
            )
        )

        exponent = (
            -safe_b_diag[:, None]
            * (fs[:, None] - f_space) ** 2
            / (2.0 * safe_variance)
        )

        log_inner_sum = logsumexp(
            log_amplitude + exponent,
            axis=0,
        )

        safe_C_sum = jnp.maximum(
            jnp.sum(
                jnp.nan_to_num(C)
            ),
            0.0,
        )

        log_constant = jnp.log(
            2.0
            * (
                safe_variance
                + safe_C_sum
            )
        )

        p_space = jnp.exp(
            log_constant + log_inner_sum
        )

        p_spec_out = jnp.column_stack(
            (
                f_space,
                p_space,
            )
        )

    # ------------------------------------------------------------------
    # Global likelihood
    # ------------------------------------------------------------------
    if calc_glob_ll:
        assert projection_hessian is not None
        assert mean_sq_data is not None
        assert mean_sq_proj is not None

        R_delta = float(
            jnp.max(jnp.abs(d))
        )
        R_sigma = float(
            jnp.max(jnp.abs(d))
        )

        log_R_delta = jnp.maximum(
            jnp.log(R_delta),
            1e-12,
        )
        log_R_sigma = jnp.maximum(
            jnp.log(R_sigma),
            1e-12,
        )

        mean_dt = float(
            jnp.mean(jnp.diff(t))
        )
        duration = float(t[-1] - t[0])

        R_gamma = (
            0.5 / mean_dt
        ) * duration

        b = (
            -m / 2.0
        ) * projection_hessian

        global_eigenvalues, _ = jnp.linalg.eigh(b)
        global_eigenvalues = jnp.maximum(
            global_eigenvalues,
            1e-12,
        )

        factor = (
            (m / 2.0) * jnp.log(2.0 * jnp.pi)
            - 0.5
            * jnp.sum(
                jnp.log(global_eigenvalues)
            )
            - m * jnp.log(R_gamma)
        )

        delta_term = (
            jsp.gammaln(m / 2.0)
            - jnp.log(2.0 * log_R_delta)
            - (m / 2.0)
            * jnp.log(
                (m * mean_sq_proj) / 2.0
            )
        )

        sigma_term = (
            jsp.gammaln(
                (N - m - r) / 2.0
            )
            - jnp.log(2.0 * log_R_sigma)
            - ((N - m - r) / 2.0)
            * jnp.log(
                (
                    N * mean_sq_data
                    - m * mean_sq_proj
                )
                / 2.0
            )
        )

        gamma_term = (
            -(2.0 * r)
            * jnp.log(R_gamma)
        )

        glob_ll_out = (
            delta_term
            + sigma_term
            + gamma_term
            + factor
        )

    return StatisticsResult(
        log_prob=log_prob_out,
        variance=variance_out,
        SNR=snr_out,
        p_spec=p_spec_out,
        glob_LL=glob_ll_out,
        cov_mat=cov_mat_out,
        fs=fs,
        ks=ks,
        f_unc=f_unc_out,
        k_unc=k_unc_out,
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

    def run_nuts(
        self,
        f_bw: float | ArrayLike,
        k_bw: float | ArrayLike,
        W: int = 1_000,
        S: int = 2_000,
        seed: int = 0,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        prior_n_std: float = 5.0,
        unbounded: bool = False,
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
                unbounded,
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


    def run_grid_search(
        self,
        min_f: float,
        max_f: float,
        min_k: float,
        max_k: float,
        f_points: int = 500,
        k_points: int = 500,
        signals: int = 1,
        apply_bandpass: bool = True,
        bandpass_order: int = 6,
        selection: Literal["best", "worst"] = "best",
        grid_batch_size: int = 4_096,
        progress_bar: bool = True,
        progress_desc: str | None = None,
        progress_position: int | None = None,
        fft_points: int = 8192,
        diagnostics: bool = False,
    ) -> BATSResult:
        """Find and subtract single signals using a frequency/decay grid.

        For each requested signal:

        1. Evaluate the one-signal Bretthorst probability over the complete
           frequency/decay-rate grid.
        2. Select either the best or worst grid point.
        3. Calculate that signal's least-squares model.
        4. Subtract the model from the working data.
        5. Repeat using the residual.

        Parameters
        ----------
        min_f, max_f
            Inclusive frequency-grid limits in Hz.
        min_k, max_k
            Inclusive decay-rate-grid limits.
        f_points, k_points
            Number of points along each grid axis.
        signals
            Number of sequential signals to find and subtract.
        apply_bandpass
            If True, apply a Butterworth bandpass from ``min_f`` to
            ``max_f`` before beginning the grid search.
        bandpass_order
            Butterworth filter order.
        selection
            ``"best"`` selects the largest log probability. ``"worst"``
            selects the smallest log probability, which can be used to
            sample the noise floor.
        grid_batch_size
            Number of frequency/decay pairs evaluated simultaneously.
            Larger values may be faster but use more device memory.
        progress_bar
            Whether to display a tqdm progress bar.
        progress_desc, progress_position
            Progress-bar controls matching ``run_nuts``.

        Returns
        -------
        BATSResult
            ``fs`` and ``ks`` contain one selected point per sequential
            grid search.
        """
        if not np.isfinite(min_f) or not np.isfinite(max_f):
            raise ValueError("min_f and max_f must be finite")
        if not np.isfinite(min_k) or not np.isfinite(max_k):
            raise ValueError("min_k and max_k must be finite")
        if min_f >= max_f:
            raise ValueError(
                f"min_f ({min_f}) must be less than max_f ({max_f})"
            )
        if min_k < 0:
            raise ValueError(f"min_k must be non-negative, got {min_k}")
        if min_k > max_k:
            raise ValueError(
                f"min_k ({min_k}) must be <= max_k ({max_k})"
            )
        if f_points < 2:
            raise ValueError(f"f_points must be >= 2, got {f_points}")
        if k_points < 2:
            raise ValueError(f"k_points must be >= 2, got {k_points}")
        if signals < 1:
            raise ValueError(f"signals must be >= 1, got {signals}")
        if bandpass_order < 1:
            raise ValueError(
                f"bandpass_order must be >= 1, got {bandpass_order}"
            )
        if grid_batch_size < 1:
            raise ValueError(
                f"grid_batch_size must be >= 1, got {grid_batch_size}"
            )
        if selection not in {"best", "worst"}:
            raise ValueError(
                "selection must be either 'best' or 'worst', "
                f"got {selection!r}"
            )
        if self.d.shape[0] <= 2:
            raise ValueError(
                "At least three data samples are required for a "
                "single-signal Bretthorst grid search"
            )
        if fft_points < 2:
            raise ValueError(
                f"fft_points must be >= 2, got {fft_points}"
            )

        diagnostics_path: Path | None = None

        if diagnostics:
            diagnostics_path = _create_diagnostics_directory()

        if apply_bandpass:
            working_data = _butter_bandpass_filter(
                self.t,
                self.d,
                min_f,
                max_f,
                bandpass_order,
            )
        else:
            working_data = self.d.copy()

        # Build the Cartesian grid. With indexing="ij", each frequency
        # is paired with every decay rate.
        f_space = jnp.linspace(
            min_f,
            max_f,
            int(f_points),
            dtype=jnp.float64,
        )
        k_space = jnp.exp(
        jnp.linspace(
            jnp.log(min_k),
            jnp.log(max_k),
            int(k_points),
            dtype=jnp.float64,
        )
    )
        f_grid, k_grid = jnp.meshgrid(
            f_space,
            k_space,
            indexing="ij",
        )
        flat_f = jnp.ravel(f_grid)
        flat_k = jnp.ravel(k_grid)

        n_grid = int(flat_f.shape[0])
        n_batches = (n_grid + grid_batch_size - 1) // grid_batch_size
        padded_size = n_batches * grid_batch_size
        padding = padded_size - n_grid

        # Pad to equal-size batches so the grid evaluator only needs one
        # JAX compilation. Padded probabilities are discarded afterward.
        if padding:
            flat_f_padded = jnp.pad(
                flat_f,
                (0, padding),
                mode="edge",
            )
            flat_k_padded = jnp.pad(
                flat_k,
                (0, padding),
                mode="edge",
            )
        else:
            flat_f_padded = flat_f
            flat_k_padded = flat_k

        frequency_batches = flat_f_padded.reshape(
            n_batches,
            grid_batch_size,
        )
        decay_batches = flat_k_padded.reshape(
            n_batches,
            grid_batch_size,
        )

        found_fs: list[jax.Array] = []
        found_ks: list[jax.Array] = []
        selected_log_probs: list[jax.Array] = []
        residual_norms: list[jax.Array] = []

        if progress_desc is None:
            progress_desc = progress_bar_label(1, signals, signals)

        def perform_search(bar: Any | None = None) -> None:
            nonlocal working_data

            for signal_index in range(signals):
                # Preserve the current residual for this iteration's
                # before/after Fourier diagnostic.
                data_before = working_data

                batched_probabilities = _evaluate_single_signal_grid(
                    self.t,
                    data_before,
                    frequency_batches,
                    decay_batches,
                )
                probabilities = jnp.ravel(batched_probabilities)[:n_grid]

                if selection == "best":
                    selected_index = jnp.argmax(probabilities)
                else:
                    selected_index = jnp.argmin(probabilities)

                # Synchronize the selected index before indexing and plotting.
                selected_index_int = int(selected_index)

                selected_f = flat_f[selected_index_int]
                selected_k = flat_k[selected_index_int]
                selected_probability = probabilities[selected_index_int]

                found_fs.append(selected_f)
                found_ks.append(selected_k)
                selected_log_probs.append(selected_probability)

                selected_model = _single_signal_model(
                    self.t,
                    data_before,
                    selected_f,
                    selected_k,
                )
                data_after = data_before - selected_model

                # Diagnostics are generated before advancing to the next
                # residual. Converting to NumPy synchronizes JAX here only
                # when diagnostics are explicitly enabled.
                if diagnostics_path is not None:
                    diagnostic_number = signal_index + 1

                    selected_f_float = float(selected_f)
                    selected_k_float = float(selected_k)
                    selected_probability_float = float(selected_probability)

                    _plot_timeseries_diagnostic(
                        path=(
                            diagnostics_path
                            / (
                                f"signal_{diagnostic_number:03d}"
                                "_timeseries_before_after.png"
                            )
                        ),
                        t=self.t,
                        data_before=data_before,
                        data_after=data_after,
                        model=selected_model,
                        selected_f=float(selected_f),
                        selected_k=float(selected_k),
                        signal_number=diagnostic_number,
                    )

                    _plot_grid_probability_diagnostic(
                        path=(
                            diagnostics_path
                            / (
                                f"signal_{diagnostic_number:03d}"
                                "_grid_probability.png"
                            )
                        ),
                        f_space=f_space,
                        k_space=k_space,
                        probabilities=probabilities,
                        selected_f=selected_f_float,
                        selected_k=selected_k_float,
                        selected_probability=selected_probability_float,
                        signal_number=diagnostic_number,
                        selection=selection,
                    )

                    _plot_fourier_diagnostic(
                        path=(
                            diagnostics_path
                            / (
                                f"signal_{diagnostic_number:03d}"
                                "_fourier_before_after.png"
                            )
                        ),
                        t=self.t,
                        data_before=data_before,
                        data_after=data_after,
                        selected_f=selected_f_float,
                        min_f=min_f,
                        max_f=max_f,
                        signal_number=diagnostic_number,
                        fft_points=fft_points,
                    )

                working_data = data_after
                residual_norms.append(jnp.linalg.norm(working_data))

                if bar is not None:
                    bar.set_postfix_str(
                        f"{selection} "
                        f"{signal_index + 1}/{signals}: "
                        f"f={float(selected_f):.6g}, "
                        f"k={float(selected_k):.6g}",
                        refresh=False,
                    )
                    bar.update(1)

        if progress_bar:
            with positioned_tqdm(progress_desc, progress_position):
                bar = tqdm.auto.tqdm(
                    total=signals,
                    desc=progress_desc,
                    position=progress_position,
                    leave=progress_position is None,
                    dynamic_ncols=True,
                    unit="signal",
                )
                try:
                    perform_search(bar)
                finally:
                    bar.close()
        else:
            perform_search()

        result_fs = jnp.stack(found_fs)
        result_ks = jnp.stack(found_ks)

        extras = {
            "selection": selection,
            "selected_log_prob": jnp.stack(selected_log_probs),
            "residual_norm": jnp.stack(residual_norms),
            "residual": working_data,
            "bandpass_applied": apply_bandpass,
            "bandpass_order": int(bandpass_order),
            "f_space": f_space,
            "k_space": k_space,
            "f_points": int(f_points),
            "k_points": int(k_points),
            "grid_batch_size": int(grid_batch_size),
            "fft_points": int(fft_points),
            "diagnostics": bool(diagnostics),
            "diagnostics_dir": (
                str(diagnostics_path)
                if diagnostics_path is not None
                else None
            ),
        }

        return BATSResult(
            fs=result_fs,
            ks=result_ks,
            extras=extras,
        )
    