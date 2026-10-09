# Internal Python libraries
from collections import defaultdict
from collections.abc import Mapping

# External libraries
import jax.numpy as jnp
import numpy as np
from scipy import signal

# Repository files
import rcrpy


def filter(t, d, f_min, f_max):
    sample_rate = 1 / np.mean(np.diff(t))
    sos = signal.butter(2, [f_min, f_max], btype='bandpass', fs=sample_rate, output='sos')

    return signal.sosfiltfilt(sos, d)

def get_best_signal(probability_surface):
    f_space, k_space, log_prob_space = probability_surface

    best_index = jnp.unravel_index(
        jnp.argmax(log_prob_space),
        log_prob_space.shape,
    )

    f_index, k_index = best_index

    return f_space[f_index], k_space[k_index]

def unpack_signals(signals):
    signals = jnp.asarray(signals)

    if signals.ndim == 1:
        if signals.shape[0] != 2:
            raise ValueError(
                f"A single signal must have shape (2,), got {signals.shape}"
            )
        signals = signals[None, :]

    elif signals.ndim == 2:
        if signals.shape[1] != 2:
            raise ValueError(
                f"Signals must have shape (n_signals, 2), got {signals.shape}"
            )

    else:
        raise ValueError(
            f"Signals must have shape (2,) or (n_signals, 2), "
            f"got {signals.shape}"
        )

    fs = signals[:, 0]
    ks = signals[:, 1]

    return fs, ks

def get_signal_bounds(
    signals,
    signal_space,
    f_fraction=1 / 8,
    log_k_fraction=1 / 8,
):
    fs, ks = unpack_signals(signals)

    fs = jnp.asarray(fs)
    ks = jnp.asarray(ks)

    if signal_space.f_max <= signal_space.f_min:
        raise ValueError("signal_space.f_max must exceed f_min")

    if signal_space.k_min <= 0:
        raise ValueError("signal_space.k_min must be positive")

    if signal_space.k_max <= signal_space.k_min:
        raise ValueError("signal_space.k_max must exceed k_min")

    if bool(jnp.any(fs < signal_space.f_min)) or bool(
        jnp.any(fs > signal_space.f_max)
    ):
        raise ValueError(
            "Signal frequencies must lie inside signal_space."
        )

    frequency_halfwidth = (
        signal_space.f_max - signal_space.f_min
    ) / f_fraction

    log_k_space_min = jnp.log(signal_space.k_min)
    log_k_space_max = jnp.log(signal_space.k_max)

    log_k_halfwidth = (
        log_k_space_max - log_k_space_min
    ) / log_k_fraction

    frequency_min = jnp.maximum(
        fs - frequency_halfwidth,
        signal_space.f_min,
    )
    frequency_max = jnp.minimum(
        fs + frequency_halfwidth,
        signal_space.f_max,
    )

    log_ks = jnp.log(ks)

    log_k_min = jnp.maximum(
        log_ks - log_k_halfwidth,
        log_k_space_min,
    )
    log_k_max = jnp.minimum(
        log_ks + log_k_halfwidth,
        log_k_space_max,
    )

    # Store physical decay-rate bounds.
    decay_rate_min = jnp.exp(log_k_min)
    decay_rate_max = jnp.exp(log_k_max)

    return jnp.stack(
        (
            frequency_min,
            frequency_max,
            decay_rate_min,
            decay_rate_max,
        ),
        axis=-1,
    )

def is_signal_detected(probability_surface, log_prob=None, n=5):
    _, _, log_prob_space = probability_surface
    log_prob_space = log_prob_space.ravel()

    if log_prob is not None:
        log_prob_space = jnp.concatenate(
            [log_prob_space, jnp.atleast_1d(log_prob)]
        )

    r = rcrpy.RCR(rcrpy.RejectionTech.ES_MODE_DL)
    r.perform_rejection(log_prob_space)

    mu = r.result.mu
    threshold = mu + n * r.result.sigma_above

    above_n_std = log_prob_space >= threshold

    if len(log_prob_space[above_n_std]) > 0:
        return True
    else:
        return False

def unpack_signal_results(results):
    signal_values = defaultdict(list)

    if isinstance(results, Mapping):
        for signal_index, entries in results.items():
            for entry in entries:
                if isinstance(entry, Mapping):
                    frequency, decay_rate = entry["result"]
                else:
                    # Supports (task_index, (frequency, decay_rate))
                    _, (frequency, decay_rate) = entry

                signal_values[signal_index].append(
                    (frequency, decay_rate)
                )

    else:
        for signal_index, (frequency, decay_rate) in results:
            signal_values[signal_index].append(
                (frequency, decay_rate)
            )

    averaged_signals = []

    for signal_index in sorted(signal_values):
        signals = signal_values[signal_index]

        average_frequency = sum(
            frequency for frequency, _ in signals
        ) / len(signals)

        average_decay_rate = sum(
            decay_rate for _, decay_rate in signals
        ) / len(signals)

        averaged_signals.append(
            (average_frequency, average_decay_rate)
        )

    return averaged_signals

def get_variance_break(
    variances,
    confidence_threshold=0.90,
    log_values=True,
):
    y = np.asarray(variances, dtype=float).reshape(-1)

    if y.size < 5:
        return None

    if not np.all(np.isfinite(y)):
        raise ValueError("variances must contain only finite values")

    if log_values:
        if np.any(y <= 0):
            raise ValueError(
                "variances must be positive when log_values=True"
            )
        y = np.log10(y)

    differences = np.diff(y)
    x = np.arange(differences.size, dtype=float)

    def line_sse(x_segment, values):
        if values.size <= 1:
            return 0.0

        coefficients = np.polyfit(x_segment, values, deg=1)
        residuals = values - np.polyval(coefficients, x_segment)
        return float(residuals @ residuals)

    # No-break model.
    one_line_sse = line_sse(x, differences)

    if one_line_sse <= np.finfo(float).eps:
        return None

    candidates = []

    # A detectable shared break must leave at least two original points
    # on each side. Thus, valid indices are 1 through len(y) - 2.
    for break_index in range(1, len(y) - 1):
        left_sse = line_sse(
            x[:break_index],
            differences[:break_index],
        )
        right_sse = line_sse(
            x[break_index:],
            differences[break_index:],
        )

        two_line_sse = left_sse + right_sse
        confidence = 1.0 - two_line_sse / one_line_sse

        candidates.append(
            (confidence, break_index)
        )

    best_confidence, best_index = max(candidates)

    if best_confidence < confidence_threshold:
        return None

    return best_index


def _get_eigenvalue_scale(eigenvalues):
    """Compute a spectral scale independently for each matrix."""
    scale = jnp.max(
        jnp.abs(eigenvalues),
        axis=-1,
        keepdims=True,
    )

    # Relative scaling is undefined for an exactly zero matrix.
    return jnp.where(
        scale > 0,
        scale,
        jnp.ones_like(scale),
    )


def get_g_eigendecomposition(
    matrix,
    tolerance=1e-10,
):
    matrix = jnp.asarray(matrix)
    matrix = 0.5 * (
        matrix
        + jnp.swapaxes(jnp.conj(matrix), -1, -2)
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)

    scale = _get_eigenvalue_scale(eigenvalues)
    eigenvalue_floor = tolerance * scale

    eigenvalues = jnp.maximum(
        eigenvalues,
        eigenvalue_floor,
    )

    return eigenvalues, eigenvectors


def get_h_eigendecomposition(
    matrix,
    ridge=1e-6,
):
    """Return an eigendecomposition of a positive-definite shifted Hessian."""
    matrix = jnp.asarray(matrix)
    matrix = 0.5 * (
        matrix
        + jnp.swapaxes(jnp.conj(matrix), -1, -2)
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)

    scale = _get_eigenvalue_scale(eigenvalues)
    relative_ridge = ridge * scale

    minimum_eigenvalue = jnp.min(
        eigenvalues,
        axis=-1,
        keepdims=True,
    )

    # Apply one scalar diagonal shift per matrix. If the Hessian is
    # indefinite, first eliminate its negative curvature and then add
    # the requested relative ridge.
    diagonal_shift = (
        jnp.maximum(-minimum_eigenvalue, 0.0)
        + relative_ridge
    )

    eigenvalues = eigenvalues + diagonal_shift

    return eigenvalues, eigenvectors