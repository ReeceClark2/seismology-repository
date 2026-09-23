from collections.abc import Mapping
from collections import defaultdict

import jax.numpy as jnp
import numpy as np
from scipy import signal

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


def get_snr_index_break(snrs):
    heuristic = []

    index = heuristic.index(max(heuristic))

    return index
