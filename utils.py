import csv
from collections.abc import Mapping
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np
from scipy import signal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import LogLocator, FuncFormatter
from matplotlib.patches import Polygon

import rcrpy


def filter(t, d, f_min, f_max):
    sample_rate = 1 / np.mean(np.diff(t))
    sos = signal.butter(4, [f_min, f_max], btype='bandpass', fs=sample_rate, output='sos')

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


def plot_time_series(path, t, d, title, model=None):
    if model is None:
        # Single plot when no model is provided
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(t, d, color="black", label="Data", linewidth=0.6)
        ax.set_xlim(min(t), max(t))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Intensity")
        ax.set_title(title)
        ax.legend()

    else:
        # Two plots when a model is provided
        fig, (ax_top, ax_bottom) = plt.subplots(
            2,
            1,
            figsize=(10, 5),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # Top plot: data and model
        ax_top.plot(t, d, color="black", label="Data", linewidth=0.6)
        ax_top.plot(t, model, color="red", label="Model", linewidth=0.6)

        ax_top.set_ylabel("Intensity")
        ax_top.set_title(title)
        ax_top.legend()

        # Bottom plot: residual
        ax_bottom.plot(t, d - model, color="blue", label="Residual", linewidth=0.6)
        ax_bottom.set_xlabel("Time (s)")
        ax_bottom.set_ylabel("Residual")
        ax_bottom.set_xlim(min(t), max(t))
        ax_bottom.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def format_e_tick(value, position):
    power = np.log(value)
    return rf"$e^{{{power:.0f}}}$"

def plot_probability_surface(path, probability_surface, title):
    f_space, k_space, log_prob_space = probability_surface
    log_prob_space = log_prob_space.T

    fig, ax = plt.subplots(figsize=(10, 5))

    heatmap = ax.pcolormesh(
        f_space,
        k_space,
        log_prob_space,
        shading="auto",
        cmap="Grays",
    )

    ax.set_yscale("log", base=np.e)

    ax.yaxis.set_major_locator(
        LogLocator(base=np.e, subs=(1.0,))
    )

    ax.yaxis.set_major_formatter(
        FuncFormatter(format_e_tick)
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Log Probability")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def add_log_ellipse(
    ax,
    f0,
    k0,
    f_bandwidth,
    log_k_bandwidth,
    **kwargs,
):
    theta = np.linspace(0, 2 * np.pi, 200)

    f_radius = f_bandwidth / 2
    log_k0 = np.log(k0)

    # Ellipse in (frequency, log(decay rate)) coordinates
    f_values = f0 + f_bandwidth * np.cos(theta)
    log_k_values = log_k0 + log_k_bandwidth * np.sin(theta)

    # Transform back to ordinary decay-rate coordinates
    k_values = np.exp(log_k_values)

    vertices = np.column_stack((f_values, k_values))

    ellipse = Polygon(
        vertices,
        closed=True,
        **kwargs,
    )

    ax.add_patch(ellipse)
    return ellipse

def plot_signal_space(
    path,
    signals,
    title,
    signals_0=None,
    signals_bw=None,
    f_min=None,
    f_max=None,
    k_min=None,
    k_max=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    fs, ks = unpack_signals(signals)

    fs = np.asarray(fs)
    ks = np.asarray(ks)

    ax.scatter(
        fs,
        ks,
        color="red",
        label="Updated Signals",
        zorder=4,
    )

    if signals_0 is not None and signals_bw is not None:
        fs_0, ks_0 = unpack_signals(signals_0)
        fs_bw, ks_bw = unpack_signals(signals_bw)

        fs_0 = np.asarray(fs_0)
        ks_0 = np.asarray(ks_0)
        fs_bw = np.repeat(fs_bw, len(fs_0))
        ks_bw = np.repeat(ks_bw, len(ks_0))

        lengths = [len(fs_0), len(ks_0), len(fs), len(ks), len(fs_bw), len(ks_bw)]

        if len(set(lengths)) != 1:
            raise ValueError(
                "Signals, reference signals, and bandwidth arrays must "
                f"have matching lengths. Got lengths: {lengths}"
            )

        ax.scatter(
            fs_0,
            ks_0,
            color="green",
            label="Initial Signals",
            zorder=5,
        )

        for f0, k0, f, k, f_bandwidth, k_bandwidth in zip(
            fs_0,
            ks_0,
            fs,
            ks,
            fs_bw,
            ks_bw,
        ):
            add_log_ellipse(
                ax,
                f0=f0,
                k0=k0,
                f_bandwidth=f_bandwidth,
                log_k_bandwidth=k_bandwidth,
                facecolor="green",
                edgecolor="green",
                alpha=0.15,
                linewidth=1,
                zorder=1,
            )

            ax.plot(
                [f0, f],
                [k0, k],
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.7,
                zorder=2,
            )

    ax.set_yscale("log", base=np.e)

    ax.yaxis.set_major_locator(
        LogLocator(base=np.e, subs=(1.0,))
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(format_e_tick)
    )

    if f_min is not None and f_max is not None:
        halfwidth = f_max - f_min
        ax.set_xlim(f_min - halfwidth * 0.05, f_max + halfwidth * 0.05)
        ax.vlines(x=[f_min, f_max], ymin=k_min, ymax=k_max, color='black', linestyle='dashed', linewidth=0.3)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_fourier_space(path, t, d, title, f_min, f_max, f_points, model=None):
    t = np.asarray(t)
    d = np.asarray(d)

    if t.ndim != 1 or d.ndim != 1:
        raise ValueError("t and d must be one-dimensional")

    if len(t) != len(d):
        raise ValueError("t and d must have matching lengths")

    if model is not None:
        model = np.asarray(model)

        if model.ndim != 1:
            raise ValueError("model must be one-dimensional")

        if len(model) != len(t):
            raise ValueError("model must have the same length as t and d")

    if len(t) < 2:
        raise ValueError("At least two time samples are required")

    sample_interval = np.mean(np.diff(t))

    window = np.hanning(len(t))
    n_fft = max(len(t), 2 * (f_points - 1))

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=sample_interval,
    )

    normalization = np.sum(window)

    data_spectrum = np.abs(
        np.fft.rfft(d * window, n=n_fft)
    ) / normalization

    frequency_mask = (
        (frequencies >= f_min)
        & (frequencies <= f_max)
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        frequencies[frequency_mask],
        data_spectrum[frequency_mask],
        color="black",
        label="Data",
        linewidth=0.6,
    )

    if model is not None:
        residual = d - model

        model_spectrum = np.abs(
            np.fft.rfft(model * window, n=n_fft)
        ) / normalization

        residual_spectrum = np.abs(
            np.fft.rfft(residual * window, n=n_fft)
        ) / normalization

        ax.plot(
            frequencies[frequency_mask],
            model_spectrum[frequency_mask],
            color="red",
            label="Model",
            linewidth=0.6,
        )

        ax.plot(
            frequencies[frequency_mask],
            residual_spectrum[frequency_mask],
            color="blue",
            label="Residual",
            linewidth=0.6,
        )

    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def save_subband_csv(path, signals, noise_variances, snrs):
    fieldnames = [
        "Signal",
        "Frequency",
        "Decay Rate",
        "Noise Variance",
        "SNR",
    ]

    signals = list(signals)
    noise_variances = np.asarray(noise_variances).reshape(-1)
    snrs = np.asarray(snrs).reshape(-1)

    if len(signals) != len(noise_variances):
        raise ValueError(
            "The number of signals must match the number of noise variances."
        )

    if len(signals) != len(snrs):
        raise ValueError(
            "The number of signals must match the number of SNR values."
        )

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for signal_index, (signal, noise_variance, snr) in enumerate(
            zip(signals, noise_variances, snrs),
            start=1,
        ):
            frequency, decay_rate = signal

            writer.writerow(
                {
                    "Signal": signal_index,
                    "Frequency": frequency,
                    "Decay Rate": decay_rate,
                    "Noise Variance": noise_variance,
                    "SNR": snr,
                }
            )


def save_initialize_csv(path, signals_by_subband):
    fieldnames = [
        "Subband",
        "Minimum Frequency",
        "Maximum Frequency",
        "Signals",
        "Noise Variance",
        "SNR",
        "Reason",
    ]

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subband_index, subband_data in signals_by_subband.items():
            fs, _ = unpack_signals(subband_data["signals"])

            writer.writerow(
                {
                    "Subband": subband_index + 1,
                    "Minimum Frequency": subband_data["f_min"],
                    "Maximum Frequency": subband_data["f_max"],
                    "Signals": len(fs),
                    "Noise Variance": subband_data["noise_variance"],
                    "SNR": subband_data["snr"],
                    "Reason": subband_data["reason"],
                }
            )


def save_block_csv(path, signals):
    fieldnames = [
        "Signal",
        "Frequency",
        "Decay Rate"
    ]

    signals = list(signals)

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for signal_index, signal in enumerate(signals, start=1):
            frequency, decay_rate = signal

            writer.writerow(
                {
                    "Signal": signal_index,
                    "Frequency": frequency,
                    "Decay Rate": decay_rate
                }
            )



def save_report_csv(path, results, uncertainties):
    if isinstance(results, Mapping):
        results_by_signal = results
    else:
        results_by_signal = {}

        for signal_index, signal in results:
            results_by_signal.setdefault(signal_index, []).append({
                "task_index": None,
                "result": signal,
            })

    signal_indices = sorted(results_by_signal)

    if len(signal_indices) != len(uncertainties):
        raise ValueError(
            "The number of uncertainty pairs must match the number "
            "of unique signal indices."
        )

    uncertainties_by_signal = dict(
        zip(signal_indices, uncertainties)
    )

    rows = []

    for signal_index in signal_indices:
        signal_entries = results_by_signal[signal_index]
        signals = []

        for entry in signal_entries:
            if isinstance(entry, Mapping):
                signal = entry["result"]
                task_index = entry.get("task_index")
            else:
                task_index, signal = entry

            frequency, decay_rate = signal

            signals.append({
                "task_index": task_index,
                "frequency": frequency,
                "decay_rate": decay_rate,
            })

        if not signals:
            continue

        frequency_error = abs(
            max(item["frequency"] for item in signals)
            - min(item["frequency"] for item in signals)
        )

        decay_rate_error = abs(
            max(item["decay_rate"] for item in signals)
            - min(item["decay_rate"] for item in signals)
        )

        if len(signals) == 1:
            frequency_error = ""
            decay_rate_error = ""

        frequency_uncertainty, decay_rate_uncertainty = (
            uncertainties_by_signal[signal_index]
        )

        for item in signals:
            rows.append({
                "Frequency": item["frequency"],
                "Frequency Uncertainty": frequency_uncertainty,
                "Decay Rate": item["decay_rate"],
                "Decay Rate Uncertainty": decay_rate_uncertainty,
                "Frequency Error": frequency_error,
                "Decay Rate Error": decay_rate_error,
            })

    fieldnames = [
        "Frequency",
        "Frequency Uncertainty",
        "Decay Rate",
        "Decay Rate Uncertainty",
        "Frequency Error",
        "Decay Rate Error",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_report_txt(path, signal_count, noise_variance, snr):
    with open(path, "w", encoding="utf-8") as file:
        file.write(f"Signal count: {signal_count}\n")
        file.write(f"Noise variance: {noise_variance}\n")
        file.write(f"SNR: {snr}\n")
