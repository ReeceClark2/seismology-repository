# Internal Python libraries
import csv
from collections.abc import Mapping

# External libraries
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.ticker import FuncFormatter, LogLocator

# Repository files
import utils


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

def add_log_rectangle(
    ax,
    f_min,
    f_max,
    k_min,
    k_max,
    **kwargs,
):
    rectangle = Rectangle(
        (f_min, k_min),
        f_max - f_min,
        k_max - k_min,
        **kwargs,
    )

    ax.add_patch(rectangle)
    return rectangle

def plot_signal_space(
    path,
    signals,
    title,
    uncertainties=None,
    signals_0=None,
    signals_bounds=None,
    signal_space=None,
):
    fig, ax = plt.subplots(figsize=(10, 5))

    fs, ks = utils.unpack_signals(signals)

    fs = np.asarray(fs)
    ks = np.asarray(ks)

    # Draw uncertainty regions from largest to smallest so that the
    # more opaque inner ellipses remain visible.
    if uncertainties is not None:
        sigma_fs, sigma_ks = utils.unpack_signals(uncertainties)

        sigma_fs = np.asarray(sigma_fs)
        sigma_ks = np.asarray(sigma_ks)

        lengths = [len(fs), len(ks), len(sigma_fs), len(sigma_ks)]
        if len(set(lengths)) != 1:
            raise ValueError(
                "Signals and uncertainties must have matching lengths. "
                f"Got lengths: {lengths}"
            )

        if np.any(ks <= 0):
            raise ValueError(
                "Decay rates must be positive to calculate log-space uncertainties."
            )

        if np.any(sigma_fs < 0) or np.any(sigma_ks < 0):
            raise ValueError("Standard deviations must be non-negative.")

        # First-order delta-method conversion:
        # Var(log(k)) ≈ Var(k) / k**2
        sigma_log_ks = sigma_ks / ks

        # Draw largest first so the inner regions remain more visible.
        uncertainty_levels = (
            (3, 0.10),
            (2, 0.25),
            (1, 0.40),
        )

        angles = np.linspace(0.0, 2.0 * np.pi, 200)

        for f, k, sigma_f, sigma_log_k in zip(
            fs,
            ks,
            sigma_fs,
            sigma_log_ks,
        ):
            log_k = np.log(k)

            for standard_deviations, alpha in uncertainty_levels:
                ellipse_f = (
                    f
                    + standard_deviations
                    * sigma_f
                    * np.cos(angles)
                )

                ellipse_log_k = (
                    log_k
                    + standard_deviations
                    * sigma_log_k
                    * np.sin(angles)
                )

                # Convert log-decay coordinates back to the data coordinates
                # expected by the logarithmic Matplotlib axis.
                ellipse_k = np.exp(ellipse_log_k)

                ax.fill(
                    ellipse_f,
                    ellipse_k,
                    facecolor="red",
                    edgecolor="red",
                    linewidth=1,
                    alpha=alpha,
                    zorder=3,
                )

    # Keep signal markers above their uncertainty regions.
    ax.scatter(
        fs,
        ks,
        color="red",
        label="Updated Signals",
        s=5,
        zorder=4,
    )

    bounds = None

    if signals_bounds is not None:
        bounds = np.asarray(signals_bounds, dtype=float)

        if bounds.ndim != 2 or bounds.shape != (len(fs), 4):
            raise ValueError(
                "signals_bounds must have shape (n_signals, 4), with entries "
                "(f_min, f_max, k_min, k_max)"
            )

        if np.any(~np.isfinite(bounds)):
            raise ValueError(
                "signals_bounds must contain only finite values."
            )

        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError(
                "Every frequency lower bound must be less than its upper bound."
            )

        if np.any(bounds[:, 2] <= 0):
            raise ValueError(
                "Decay-rate bounds must be positive on a logarithmic axis."
            )

        if np.any(bounds[:, 2] >= bounds[:, 3]):
            raise ValueError(
                "Every decay-rate lower bound must be less than its upper bound."
            )

        for f_min, f_max, k_min, k_max in bounds:
            add_log_rectangle(
                ax,
                f_min=f_min,
                f_max=f_max,
                k_min=k_min,
                k_max=k_max,
                facecolor="green",
                edgecolor="green",
                alpha=0.15,
                linewidth=1,
                zorder=1,
            )

    if signals_0 is not None:
        fs_0, ks_0 = utils.unpack_signals(signals_0)

        fs_0 = np.asarray(fs_0)
        ks_0 = np.asarray(ks_0)

        if len(fs_0) != len(fs):
            raise ValueError(
                "signals_0 and signals must have matching lengths "
                "when plotting movement between signals."
            )

        ax.scatter(
            fs_0,
            ks_0,
            color="green",
            label="Initial Signals",
            zorder=5,
        )

        for f0, k0, f, k in zip(fs_0, ks_0, fs, ks):
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

    if signal_space is not None:
        f_min = signal_space.f_min
        f_max = signal_space.f_max
        k_min = signal_space.k_min
        k_max = signal_space.k_max

        halfwidth = f_max - f_min
        ax.set_xlim(
            f_min - halfwidth * 0.1,
            f_max + halfwidth * 0.1,
        )

        log_k_min = np.log(k_min)
        log_k_max = np.log(k_max)

        log_k_halfwidth = log_k_max - log_k_min
        log_padding = 0.1 * log_k_halfwidth

        ax.set_ylim(
            np.exp(log_k_min - log_padding),
            np.exp(log_k_max + log_padding),
        )

        xmin, xmax = ax.get_xlim()
        ax.hlines(
            y=[k_min, k_max],
            xmin=f_min,
            xmax=f_max,
            color="black",
            linestyle="dashed",
            linewidth=0.3,
        )
        ax.set_xlim(xmin, xmax)

        ymin, ymax = ax.get_ylim()
        ax.vlines(
            x=[f_min, f_max],
            ymin=k_min,
            ymax=k_max,
            color="black",
            linestyle="dashed",
            linewidth=0.3,
        )
        ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

def plot_fourier_space(
    path,
    t,
    d,
    title,
    f_min,
    f_max,
    f_points,
    signals=None,
    model=None,
):
    t = np.asarray(t)
    d = np.asarray(d)

    if t.ndim != 1 or d.ndim != 1:
        raise ValueError("t and d must be one-dimensional")

    if len(t) != len(d):
        raise ValueError("t and d must have matching lengths")

    if len(t) < 2:
        raise ValueError("At least two time samples are required")

    if f_min < 0 or f_max <= f_min:
        raise ValueError("Require 0 <= f_min < f_max.")

    if f_points < 2:
        raise ValueError("f_points must be at least 2.")

    if model is not None:
        model = np.asarray(model)

        if model.ndim != 1:
            raise ValueError("model must be one-dimensional")

        if len(model) != len(t):
            raise ValueError(
                "model must have the same length as t and d"
            )

    dt_values = np.diff(t)

    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError(
            "The FFT requires uniformly sampled time values."
        )

    sample_interval = float(dt_values[0])
    sample_rate = 1.0 / sample_interval
    nyquist = 0.5 * sample_rate

    if f_max > nyquist:
        raise ValueError(
            f"f_max={f_max} exceeds the Nyquist frequency "
            f"{nyquist}."
        )

    # Exactly f_points frequencies in [f_min, f_max].
    target_frequencies = np.linspace(
        f_min,
        f_max,
        f_points,
    )

    # Native FFT frequency spacing should be fine enough for interpolation.
    target_spacing = (
        f_max - f_min
    ) / (f_points - 1)

    required_n_fft = int(
        np.ceil(
            1.0
            / (sample_interval * target_spacing)
        )
    )

    n_fft = max(len(t), required_n_fft)

    frequencies = np.fft.rfftfreq(
        n_fft,
        d=sample_interval,
    )

    def interpolated_spectrum(x):
        spectrum = np.abs(
            np.fft.rfft(x, n=n_fft)
        )

        return np.interp(
            target_frequencies,
            frequencies,
            spectrum,
        )

    data_spectrum = interpolated_spectrum(d)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        target_frequencies,
        data_spectrum,
        color="black",
        label="Data",
        linewidth=0.6,
    )

    if model is not None:
        residual = d - model

        model_spectrum = interpolated_spectrum(model)
        residual_spectrum = interpolated_spectrum(residual)

        ax.plot(
            target_frequencies,
            model_spectrum,
            color="red",
            label="Model",
            linewidth=0.6,
        )

        ax.plot(
            target_frequencies,
            residual_spectrum,
            color="blue",
            label="Residual",
            linewidth=0.6,
        )

    ax.set_xlim(f_min, f_max)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    if signals is not None:
        fs, _ = utils.unpack_signals(signals)
        fs = np.asarray(fs).ravel()

        # Capture the limits established by the spectra before adding
        # the vertical lines.
        y_min, y_max = ax.get_ylim()

        ax.vlines(
            fs,
            ymin=y_min,
            ymax=y_max,
            color="red",
            linestyle="--",
            linewidth=0.4,
            label="Signals",
        )

        # Explicitly restore the original limits so the lines do not
        # affect the y-axis scaling.
        ax.set_ylim(y_min, y_max)

    ax.legend()

    fig.tight_layout()
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)

def save_subband_csv(path, signals, noise_variances, snrs, glob_lls):
    fieldnames = [
        "Signal",
        "Frequency",
        "Decay Rate",
        "Noise Variance",
        "SNR",
        "Global Likelihood",
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

        for signal_index, (signal, noise_variance, snr, glob_ll) in enumerate(
            zip(signals, noise_variances, snrs, glob_lls),
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
                    "Global Likelihood": glob_ll
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
        "Global Likelihood",
        "Reason",
    ]

    with open(path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for subband_index, subband_data in signals_by_subband.items():
            fs, _ = utils.unpack_signals(subband_data["signals"])

            writer.writerow(
                {
                    "Subband": subband_index + 1,
                    "Minimum Frequency": subband_data["f_min"],
                    "Maximum Frequency": subband_data["f_max"],
                    "Signals": len(fs),
                    "Noise Variance": subband_data["noise_variance"],
                    "SNR": subband_data["snr"],
                    "Global Likelihood": subband_data["glob_ll"],
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

def save_signals_csv(path, signals, amplitudes, phases, uncertainties):
    signals = list(signals)
    amplitudes = list(amplitudes)
    phases = list(phases)
    uncertainties = list(uncertainties)

    if len(signals) != len(uncertainties):
        raise ValueError(
            "The number of uncertainty pairs must match the number of signals."
        )

    rows = []

    for signal, amplitude, phase, uncertainty in zip(signals, amplitudes, phases, uncertainties):
        try:
            frequency, decay_rate = signal
            frequency_uncertainty, decay_rate_uncertainty = uncertainty
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Each signal must be a `(frequency, decay_rate)` pair, "
                "and each uncertainty must be a "
                "`(frequency_uncertainty, decay_rate_uncertainty)` pair."
            ) from exc

        rows.append({
            "Amplitude": amplitude,
            "Phase": phase,
            "Frequency": frequency,
            "Frequency Uncertainty": frequency_uncertainty,
            "Decay Rate": decay_rate,
            "Decay Rate Uncertainty": decay_rate_uncertainty,
        })

    fieldnames = [
        "Amplitude",
        "Phase",
        "Frequency",
        "Frequency Uncertainty",
        "Decay Rate",
        "Decay Rate Uncertainty",
    ]

    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

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

def plot_covariance_and_correlation(
    path,
    cov_mat,
    signals,
):
    """
    Plot raw covariance, quadrant-normalized covariance, and correlation.

    The covariance matrix is assumed to use the flattened parameter order:

        [f0, k0, f1, k1, ...]

    For plotting, it is reordered to:

        [f0, f1, ..., fn, k0, k1, ..., kn]
    """
    covariance = np.asarray(cov_mat, dtype=float)
    signals = np.asarray(signals)

    if signals.ndim == 1:
        if signals.shape != (2,):
            raise ValueError(
                f"A single signal must have shape (2,), got {signals.shape}"
            )
        signals = signals[None, :]

    if signals.ndim != 2 or signals.shape[1] != 2:
        raise ValueError(
            "signals must have shape (n_signals, 2), "
            f"got {signals.shape}"
        )

    if covariance.ndim != 2:
        raise ValueError(
            f"cov_mat must be two-dimensional, got {covariance.shape}"
        )

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(
            f"cov_mat must be square, got {covariance.shape}"
        )

    n_signals = len(signals)
    parameter_count = 2 * n_signals

    if covariance.shape != (parameter_count, parameter_count):
        raise ValueError(
            "cov_mat shape must match the flattened signal parameters. "
            f"Expected {(parameter_count, parameter_count)}, "
            f"got {covariance.shape}"
        )

    if not np.all(np.isfinite(covariance)):
        raise ValueError("cov_mat must contain only finite values.")

    # Remove small numerical asymmetries.
    covariance = 0.5 * (covariance + covariance.T)

    # Original order:
    # [f0, k0, f1, k1, ...]
    #
    # Plotting order:
    # [f0, f1, ..., fn, k0, k1, ..., kn]
    permutation = np.concatenate((
        2 * np.arange(n_signals),
        2 * np.arange(n_signals) + 1,
    ))

    covariance = covariance[
        np.ix_(permutation, permutation)
    ]

    labels = (
        [
            rf"$f_{index}$"
            for index in range(n_signals)
        ]
        + [
            rf"$k_{index}$"
            for index in range(n_signals)
        ]
    )

    # ------------------------------------------------------------------
    # Create quadrant-normalized covariance.
    #
    # Matrix layout:
    #
    #              frequencies       decay rates
    #       frequencies       FF                FK
    #       decay rates       KF                KK
    #
    # Each quadrant is independently divided by its largest absolute
    # value. This preserves relative structure within each quadrant.
    # ------------------------------------------------------------------
    normalized_covariance = np.empty_like(covariance)

    frequency_indices = np.arange(n_signals)
    decay_indices = np.arange(n_signals, parameter_count)

    quadrants = (
        (frequency_indices, frequency_indices),  # FF
        (frequency_indices, decay_indices),      # FK
        (decay_indices, frequency_indices),      # KF
        (decay_indices, decay_indices),           # KK
    )

    for row_indices, column_indices in quadrants:
        block = covariance[
            np.ix_(row_indices, column_indices)
        ]

        scale = np.max(np.abs(block))

        if scale > 0:
            normalized_block = block / scale
        else:
            normalized_block = np.zeros_like(block)

        normalized_covariance[
            np.ix_(row_indices, column_indices)
        ] = normalized_block

    # Re-enforce symmetry to remove tiny numerical differences between
    # the independently normalized off-diagonal blocks.
    normalized_covariance = (
        0.5
        * (normalized_covariance + normalized_covariance.T)
    )

    # ------------------------------------------------------------------
    # Calculate the ordinary correlation matrix from the raw covariance.
    # ------------------------------------------------------------------
    variances = np.diag(covariance)

    if np.any(variances < 0):
        raise ValueError(
            "Covariance matrix contains negative diagonal entries."
        )

    standard_deviations = np.sqrt(variances)
    denominator = np.outer(
        standard_deviations,
        standard_deviations,
    )

    correlation = np.full_like(covariance, np.nan)

    valid = denominator > 0
    correlation[valid] = covariance[valid] / denominator[valid]

    positive_variance = variances > 0
    diagonal_indices = np.diag_indices_from(correlation)
    correlation[diagonal_indices] = np.where(
        positive_variance,
        1.0,
        np.nan,
    )

    # ------------------------------------------------------------------
    # Plot all three matrices.
    # ------------------------------------------------------------------
    raw_covariance_scale = np.max(np.abs(covariance))

    if raw_covariance_scale == 0:
        raw_covariance_scale = 1.0

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5.5),
        constrained_layout=True,
    )

    raw_image = axes[0].imshow(
        covariance,
        cmap="RdBu_r",
        vmin=-raw_covariance_scale,
        vmax=raw_covariance_scale,
        interpolation="nearest",
    )

    normalized_image = axes[1].imshow(
        normalized_covariance,
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    correlation_image = axes[2].imshow(
        np.ma.masked_invalid(correlation),
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )

    titles = (
        "Raw Covariance",
        "Quadrant-Normalized Covariance",
        "Correlation",
    )

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xticks(np.arange(parameter_count))
        ax.set_yticks(np.arange(parameter_count))
        ax.set_xticklabels(labels, rotation=90)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Parameter")
        ax.set_ylabel("Parameter")

    raw_colorbar = fig.colorbar(
        raw_image,
        ax=axes[0],
        fraction=0.046,
        pad=0.04,
    )
    raw_colorbar.set_label("Covariance")

    normalized_colorbar = fig.colorbar(
        normalized_image,
        ax=axes[1],
        fraction=0.046,
        pad=0.04,
    )
    normalized_colorbar.set_label("Scaled covariance")

    correlation_colorbar = fig.colorbar(
        correlation_image,
        ax=axes[2],
        fraction=0.046,
        pad=0.04,
    )
    correlation_colorbar.set_label("Correlation")

    fig.savefig(path, dpi=300)
    plt.close(fig)
    