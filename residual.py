from bats import get_model, get_statistics
from initial_conditions import observed_data
from obspy.core import UTCDateTime

import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# =====================================================================
# Configuration
# =====================================================================

INPUT_CSV = (
    "C:/Users/starb/Downloads/dracula_output_3_4_KIP/N032_signals.csv"
)

network = "II"
station = "AAK"
location = "00"
channel = "LHZ"
stream_index = 0

start_time = UTCDateTime("2025-07-29T23:24:50")
end_time = UTCDateTime("2025-08-06T05:24:50")

min_f = 0.0030
max_f = 0.0040

# Data samples retained after downloading and filtering.
start_index = 200
stop_index = 10000

# Zero-padding factor for the displayed FFT frequency grid.
zero_padding_factor = 2

# get_statistics() can be computationally expensive.
calculate_statistics = True


# =====================================================================
# FFT
# =====================================================================

def one_sided_fft_amplitude(
    x,
    sampling_frequency,
    nfft=None,
):
    """
    Calculate a mean-removed, one-sided FFT amplitude spectrum.

    For positive frequencies, the returned amplitude is

        2 * abs(FFT) / N,

    where N is the number of original time-domain samples.
    """
    x = np.asarray(x, dtype=float).squeeze()

    if x.ndim != 1:
        raise ValueError(
            f"Expected one-dimensional input, got shape {x.shape}"
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Signal contains NaN or infinite values"
        )

    n_samples = x.size

    if n_samples < 2:
        raise ValueError(
            "Signal must contain at least two samples"
        )

    sampling_frequency = float(sampling_frequency)

    if sampling_frequency <= 0.0:
        raise ValueError(
            "Sampling frequency must be positive"
        )

    # Match the behavior of the original script.
    x = x - np.mean(x)

    if nfft is None:
        nfft = n_samples

    nfft = int(nfft)

    if nfft < n_samples:
        raise ValueError(
            "nfft must be at least as large as the signal"
        )

    spectrum = np.fft.rfft(
        x,
        n=nfft,
    )

    frequency = np.fft.rfftfreq(
        nfft,
        d=1.0 / sampling_frequency,
    )

    # Normalize using the number of original samples, not nfft.
    amplitude = np.abs(spectrum) / n_samples

    # One-sided amplitude normalization. Do not double DC or Nyquist.
    if nfft % 2 == 0:
        amplitude[1:-1] *= 2.0
    else:
        amplitude[1:] *= 2.0

    return frequency, amplitude


# =====================================================================
# Recover the amplitude of each damped mode
# =====================================================================

def fit_mode_coefficients(
    t,
    d,
    frequencies,
    decay_rates,
):
    """
    Fit the linear coefficients of fixed-frequency, fixed-decay modes.

    The model is

        model(t) = sum_i exp(-k_i t) * [
            a_i cos(2*pi*f_i*t)
            + b_i sin(2*pi*f_i*t)
        ].

    Frequencies and decay rates are held fixed. Only a_i and b_i are
    determined by linear least squares.
    """
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    frequencies = np.asarray(
        frequencies,
        dtype=float,
    ).ravel()

    decay_rates = np.asarray(
        decay_rates,
        dtype=float,
    ).ravel()

    if t.size != d.size:
        raise ValueError(
            "t and d must have the same length"
        )

    if frequencies.size != decay_rates.size:
        raise ValueError(
            "frequencies and decay_rates must have the same length"
        )

    if not np.all(np.isfinite(frequencies)):
        raise ValueError(
            "The frequency array contains non-finite values"
        )

    if not np.all(np.isfinite(decay_rates)):
        raise ValueError(
            "The decay-rate array contains non-finite values"
        )

    # Make the beginning of the selected record t = 0.
    tau = t - t[0]

    phase = (
        2.0
        * np.pi
        * frequencies[:, None]
        * tau[None, :]
    )

    decay = np.exp(
        -decay_rates[:, None]
        * tau[None, :]
    )

    cosine_terms = decay * np.cos(phase)
    sine_terms = decay * np.sin(phase)

    # Shape: number of samples by 2 * number of modes.
    design_matrix = np.concatenate(
        (
            cosine_terms.T,
            sine_terms.T,
        ),
        axis=1,
    )

    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design_matrix,
        d,
        rcond=None,
    )

    n_modes = frequencies.size

    cosine_coefficients = coefficients[:n_modes]
    sine_coefficients = coefficients[n_modes:]

    fitted_model = design_matrix @ coefficients

    # For
    #
    #   a cos(omega*t) + b sin(omega*t)
    #       = A cos(omega*t + phi),
    #
    # A = hypot(a, b) and phi = atan2(-b, a).
    mode_amplitudes = np.hypot(
        cosine_coefficients,
        sine_coefficients,
    )

    mode_phases = np.arctan2(
        -sine_coefficients,
        cosine_coefficients,
    )

    return {
        "model": fitted_model,
        "cosine_coefficients": cosine_coefficients,
        "sine_coefficients": sine_coefficients,
        "amplitudes": mode_amplitudes,
        "phases": mode_phases,
        "rank": rank,
        "singular_values": singular_values,
    }


# =====================================================================
# Ideal infinite-duration Lorentzians
# =====================================================================

def ideal_lorentzian_profiles(
    frequency_grid,
    mode_frequencies,
    decay_rates,
    mode_amplitudes,
    record_duration,
):
    """
    Calculate ideal infinite-duration frequency-domain profiles.

    For one mode,

        x_i(t) = A_i exp(-k_i*t) cos(2*pi*f_i*t + phi_i),

    the positive-frequency approximation to its Fourier amplitude is

        |X_i(f)| approximately
            A_i / [2 sqrt(k_i^2 + (2*pi*(f-f_i))^2)].

    After applying the one-sided FFT normalization 2/N and replacing
    the sample sum by an integral divided by dt, the plotted amplitude
    becomes

        L_A,i(f) =
            A_i / [
                T sqrt(k_i^2 + (2*pi*(f-f_i))^2)
            ],

    where T = N*dt.

    The squared profile

        L_P,i(f) = L_A,i(f)^2

    is Lorentzian.
    """
    frequency_grid = np.asarray(
        frequency_grid,
        dtype=float,
    ).ravel()

    mode_frequencies = np.asarray(
        mode_frequencies,
        dtype=float,
    ).ravel()

    decay_rates = np.asarray(
        decay_rates,
        dtype=float,
    ).ravel()

    mode_amplitudes = np.asarray(
        mode_amplitudes,
        dtype=float,
    ).ravel()

    if not (
        mode_frequencies.size
        == decay_rates.size
        == mode_amplitudes.size
    ):
        raise ValueError(
            "Mode frequency, decay-rate, and amplitude arrays "
            "must have the same length"
        )

    if record_duration <= 0.0:
        raise ValueError(
            "record_duration must be positive"
        )

    n_modes = mode_frequencies.size
    n_frequencies = frequency_grid.size

    amplitude_profiles = np.full(
        (n_modes, n_frequencies),
        np.nan,
        dtype=float,
    )

    positive_decay = decay_rates > 0.0

    if np.any(~positive_decay):
        invalid_indices = np.where(~positive_decay)[0]

        print(
            "Warning: ideal infinite-duration Lorentzians cannot be "
            "constructed for nonpositive decay rates. Skipping mode "
            f"indices: {invalid_indices.tolist()}"
        )

    if np.any(positive_decay):
        frequency_offset = (
            frequency_grid[None, :]
            - mode_frequencies[positive_decay, None]
        )

        angular_frequency_offset = (
            2.0
            * np.pi
            * frequency_offset
        )

        denominator = np.sqrt(
            decay_rates[positive_decay, None] ** 2
            + angular_frequency_offset**2
        )

        amplitude_profiles[positive_decay] = (
            mode_amplitudes[positive_decay, None]
            / (
                record_duration
                * denominator
            )
        )

    # This is the actual Lorentzian quantity.
    power_profiles = amplitude_profiles**2

    return amplitude_profiles, power_profiles


# =====================================================================
# Read frequencies and decay rates
# =====================================================================

df = pd.read_csv(INPUT_CSV)

required_columns = {
    "frequencies",
    "decay_rates",
}

missing_columns = required_columns.difference(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing CSV columns: {sorted(missing_columns)}"
    )

mode_frequencies = df[
    "frequencies"
].to_numpy(dtype=float)

decay_rates = df[
    "decay_rates"
].to_numpy(dtype=float)

if mode_frequencies.size != decay_rates.size:
    raise ValueError(
        "frequencies and decay_rates must have the same length"
    )

if mode_frequencies.size == 0:
    raise ValueError(
        "No modes were found in the input CSV"
    )

print(f"Loaded {mode_frequencies.size} modes")


# =====================================================================
# Retrieve observed data
# =====================================================================

t, observed = observed_data(
    network=network,
    station=station,
    channel=channel,
    location=location,
    stream_index=stream_index,
    start_time=start_time,
    end_time=end_time,
    min_f=min_f,
    max_f=max_f,
)

t = np.asarray(
    t[start_index:stop_index],
    dtype=float,
).ravel()

observed = np.asarray(
    observed[start_index:stop_index],
    dtype=float,
).ravel()

if t.size != observed.size:
    raise ValueError(
        "The selected time and data arrays have different lengths"
    )

if t.size < 2:
    raise ValueError(
        "At least two selected samples are required"
    )

dt_values = np.diff(t)

if np.any(dt_values <= 0.0):
    raise ValueError(
        "The time array must be strictly increasing"
    )

dt = float(np.median(dt_values))
sampling_frequency = 1.0 / dt
n_samples = observed.size

# N*dt is consistent with the normalization of an N-point DFT.
record_duration = n_samples * dt

print(f"Number of samples: {n_samples}")
print(f"Sampling interval: {dt:.8g} s")
print(f"Sampling frequency: {sampling_frequency:.8g} Hz")
print(f"Record duration used for normalization: {record_duration:.8g} s")


# =====================================================================
# Construct the BATS model and residual
# =====================================================================

# get_model() returns one array, not a tuple.
bats_model = np.asarray(
    get_model(
        t,
        observed,
        mode_frequencies,
        decay_rates,
    ),
    dtype=float,
).ravel()

if bats_model.size != observed.size:
    raise ValueError(
        "The BATS model and observed data have different lengths"
    )

residual = observed - bats_model


# =====================================================================
# Fit individual linear mode coefficients
# =====================================================================

coefficient_result = fit_mode_coefficients(
    t=t,
    d=observed,
    frequencies=mode_frequencies,
    decay_rates=decay_rates,
)

least_squares_model = coefficient_result["model"]
mode_amplitudes = coefficient_result["amplitudes"]
mode_phases = coefficient_result["phases"]

# get_model() and the direct least-squares projection should represent
# the same fitted model subspace. This checks their numerical agreement.
model_difference = bats_model - least_squares_model

relative_model_difference = (
    np.linalg.norm(model_difference)
    / max(np.linalg.norm(bats_model), np.finfo(float).eps)
)

print(
    "Relative difference between get_model() and direct "
    f"least-squares model: {relative_model_difference:.6e}"
)


# =====================================================================
# Print mode information
# =====================================================================

print()
print("Mode parameters")
print("-" * 100)

for index, (
    mode_frequency,
    decay_rate,
    mode_amplitude,
    mode_phase,
) in enumerate(
    zip(
        mode_frequencies,
        decay_rates,
        mode_amplitudes,
        mode_phases,
    ),
    start=1,
):
    if decay_rate > 0.0:
        power_hwhm = decay_rate / (2.0 * np.pi)
        power_fwhm = decay_rate / np.pi

        # The square-root Lorentzian reaches half its peak when
        # |f-f0| = sqrt(3)*k/(2*pi).
        amplitude_fwhm = (
            np.sqrt(3.0)
            * decay_rate
            / np.pi
        )
    else:
        power_hwhm = np.nan
        power_fwhm = np.nan
        amplitude_fwhm = np.nan

    print(
        f"{index:03d}: "
        f"f = {mode_frequency:.9f} Hz, "
        f"k = {decay_rate:.6e} 1/s, "
        f"A = {mode_amplitude:.6e}, "
        f"phase = {mode_phase:.6f} rad, "
        f"power FWHM = {power_fwhm:.6e} Hz, "
        f"amplitude FWHM = {amplitude_fwhm:.6e} Hz"
    )


# =====================================================================
# Calculate FFT spectra
# =====================================================================

minimum_nfft = zero_padding_factor * n_samples

nfft = 1 << int(
    np.ceil(np.log2(minimum_nfft))
)

frequency, observed_fft = one_sided_fft_amplitude(
    observed,
    sampling_frequency,
    nfft=nfft,
)

_, model_fft = one_sided_fft_amplitude(
    bats_model,
    sampling_frequency,
    nfft=nfft,
)

_, residual_fft = one_sided_fft_amplitude(
    residual,
    sampling_frequency,
    nfft=nfft,
)

frequency_mask = (
    (frequency >= min_f)
    & (frequency <= max_f)
)

plot_frequency = frequency[frequency_mask]

observed_fft_band = observed_fft[frequency_mask]
model_fft_band = model_fft[frequency_mask]
residual_fft_band = residual_fft[frequency_mask]


# =====================================================================
# Calculate ideal Lorentzians directly in frequency space
# =====================================================================

(
    lorentzian_amplitudes,
    lorentzian_powers,
) = ideal_lorentzian_profiles(
    frequency_grid=plot_frequency,
    mode_frequencies=mode_frequencies,
    decay_rates=decay_rates,
    mode_amplitudes=mode_amplitudes,
    record_duration=record_duration,
)

# Lorentzian powers may be added under an incoherent-mode assumption.
# This is not generally equal to the power of the coherent model FFT.
total_incoherent_lorentzian_power = np.nansum(
    lorentzian_powers,
    axis=0,
)


# =====================================================================
# Optional global statistics
# =====================================================================

if calculate_statistics:
    statistics = get_statistics(
        t,
        observed,
        mode_frequencies,
        decay_rates,
    )

    print()
    print(f"SNR: {float(statistics.SNR):.8g}")
    print(f"Variance: {float(statistics.variance):.8g}")
    print(f"Log probability: {float(statistics.log_prob):.8g}")


# =====================================================================
# Plot
# =====================================================================

fig, (amplitude_axis, power_axis) = plt.subplots(
    2,
    1,
    figsize=(13, 10),
    sharex=True,
)


# ---------------------------------------------------------------------
# Top panel: FFT amplitude and square-root Lorentzian profiles
# ---------------------------------------------------------------------

amplitude_axis.plot(
    plot_frequency,
    observed_fft_band,
    color="black",
    linewidth=1.1,
    label="Observed-data FFT",
)

amplitude_axis.plot(
    plot_frequency,
    model_fft_band,
    color="red",
    linewidth=1.2,
    label="Model FFT",
)

amplitude_axis.plot(
    plot_frequency,
    residual_fft_band,
    color="blue",
    linewidth=1.0,
    label="Residual FFT",
)

for mode_index, profile in enumerate(
    lorentzian_amplitudes
):
    if not np.any(np.isfinite(profile)):
        continue

    amplitude_axis.plot(
        plot_frequency,
        profile,
        color="darkorange",
        linewidth=0.8,
        alpha=0.45,
        label=(
            "Individual ideal amplitude profiles"
            if mode_index == 0
            else None
        ),
    )

amplitude_height = 1.05 * max(
    np.max(observed_fft_band),
    np.max(model_fft_band),
    np.nanmax(lorentzian_amplitudes),
)

amplitude_axis.vlines(
    x=mode_frequencies,
    ymin=0.0,
    ymax=amplitude_height,
    color="red",
    linestyle=":",
    linewidth=0.5,
    alpha=0.55,
    label="Model frequencies",
)

amplitude_axis.set_ylabel(
    "One-sided amplitude"
)

amplitude_axis.set_title(
    "FFT amplitudes and ideal infinite-duration mode profiles"
)

amplitude_axis.set_ylim(
    bottom=0.0,
)

amplitude_axis.grid(
    alpha=0.2,
)

amplitude_axis.legend(
    loc="upper right",
)


# ---------------------------------------------------------------------
# Bottom panel: squared FFT amplitude and true Lorentzian profiles
# ---------------------------------------------------------------------

observed_power = observed_fft_band**2
model_power = model_fft_band**2
residual_power = residual_fft_band**2

power_axis.plot(
    plot_frequency,
    observed_power,
    color="black",
    linewidth=1.1,
    label="Observed FFT squared amplitude",
)

power_axis.plot(
    plot_frequency,
    model_power,
    color="red",
    linewidth=1.2,
    label="Model FFT squared amplitude",
)

power_axis.plot(
    plot_frequency,
    residual_power,
    color="blue",
    linewidth=1.0,
    label="Residual FFT squared amplitude",
)

for mode_index, profile in enumerate(
    lorentzian_powers
):
    if not np.any(np.isfinite(profile)):
        continue

    power_axis.plot(
        plot_frequency,
        profile,
        color="darkorange",
        linewidth=0.8,
        alpha=0.45,
        label=(
            "Individual ideal Lorentzian powers"
            if mode_index == 0
            else None
        ),
    )

power_axis.plot(
    plot_frequency,
    total_incoherent_lorentzian_power,
    color="darkorange",
    linestyle="--",
    linewidth=1.6,
    label="Sum of ideal Lorentzian powers",
)

power_axis.set_xlabel(
    "Frequency (Hz)"
)

power_axis.set_ylabel(
    "Squared amplitude"
)

power_axis.set_title(
    "FFT squared amplitudes and ideal Lorentzian power profiles"
)

power_axis.set_xlim(
    min_f,
    max_f,
)

power_axis.set_ylim(
    bottom=0.0,
)

power_axis.grid(
    alpha=0.2,
)

power_axis.legend(
    loc="upper right",
)

fig.suptitle(
    f"{network}.{station}.{location}.{channel}: "
    "Model, residual, and ideal Lorentzian profiles",
    fontsize=14,
)

fig.tight_layout()
plt.show()
