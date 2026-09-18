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

def is_signal_detected(probability_surface):
    _, _, log_prob_space = probability_surface
    log_prob_space = log_prob_space.ravel()

    r = rcrpy.RCR(rcrpy.RejectionTech.ES_MODE_DL)
    r.perform_rejection(log_prob_space)

    n = 5
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
        ax.set_xlim(f_min, f_max)

    # Do not take the logarithm here. Matplotlib expects data values.
    if k_min is not None and k_max is not None:
        ax.set_ylim(k_min, k_max)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
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



def save_report_csv(path, results):
    # Support either the results_by_signal dictionary format or the original
    # iterable of (signal_index, signal) pairs.
    if isinstance(results, Mapping):
        results_by_signal = results
    else:
        results_by_signal = {}

        for signal_index, signal in results:
            results_by_signal.setdefault(signal_index, []).append({
                "task_index": None,
                "result": signal,
            })

    rows = []

    for signal_index, signal_entries in results_by_signal.items():
        signals = []

        for entry in signal_entries:
            # Dictionary format:
            if isinstance(entry, Mapping):
                signal = entry["result"]
                task_index = entry.get("task_index")
            else:
                # Also support entries of the form:
                # (task_index, (frequency, decay_rate))
                task_index, signal = entry

            frequency, decay_rate = signal

            signals.append({
                "task_index": task_index,
                "frequency": frequency,
                "decay_rate": decay_rate,
            })

        if not signals:
            continue

        # For repeated signal indices, the range gives the absolute
        # difference between the two most extreme values. Any middle values
        # are therefore ignored.
        frequency_error = abs(
            max(item["frequency"] for item in signals)
            - min(item["frequency"] for item in signals)
        )

        decay_rate_error = abs(
            max(item["decay_rate"] for item in signals)
            - min(item["decay_rate"] for item in signals)
        )

        # For a signal index occurring only once, there is no comparison.
        if len(signals) == 1:
            frequency_error = ""
            decay_rate_error = ""

        for item in signals:
            rows.append({
                "Frequency": item["frequency"],
                "Frequency Uncertainty": "",
                "Decay Rate": item["decay_rate"],
                "Decay Rate Uncertainty": "",
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
        