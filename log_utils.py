import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Load signal parameters
# ---------------------------------------------------------------------

df = pd.read_csv(
    r"C:/Users/starb/Downloads/batch_2/trial_10/reconcile/signals.csv"
)

As = df["Amplitude"].to_numpy()
phis = df["Phase"].to_numpy() + (np.pi / 2)
fs = df["Frequency"].to_numpy()
ks = df["Decay Rate"].to_numpy()

# Uncomment this if Phase is stored in degrees rather than radians:
# phis = np.deg2rad(phis)


# ---------------------------------------------------------------------
# Construct the time-domain model
# ---------------------------------------------------------------------

def time_model(t):
    result = np.zeros_like(t, dtype=float)

    for A, phi, f, k in zip(As, phis, fs, ks):
        result += (
            A
            * np.exp(-k * t)
            * np.cos(2 * np.pi * f * t + phi)
        )

    return result


# ---------------------------------------------------------------------
# FFT procedure matching plot_fourier_space()
# ---------------------------------------------------------------------

def interpolated_fft_spectrum(
    x,
    t,
    f_min,
    f_max,
    f_points,
):
    """
    Matches the FFT behavior in plot_fourier_space():

        spectrum = abs(rfft(x, n=n_fft))

    The native FFT spectrum is interpolated onto exactly
    f_points frequencies between f_min and f_max.
    """

    t = np.asarray(t)
    x = np.asarray(x)

    if t.ndim != 1 or x.ndim != 1:
        raise ValueError("t and x must be one-dimensional")

    if len(t) != len(x):
        raise ValueError("t and x must have matching lengths")

    if len(t) < 2:
        raise ValueError("At least two time samples are required")

    if f_min < 0 or f_max <= f_min:
        raise ValueError("Require 0 <= f_min < f_max")

    if f_points < 2:
        raise ValueError("f_points must be at least 2")

    # Verify uniform sampling
    dt_values = np.diff(t)

    if not np.allclose(dt_values, dt_values[0]):
        raise ValueError(
            "The FFT requires uniformly sampled time values."
        )

    dt = float(dt_values[0])
    sample_rate = 1.0 / dt
    nyquist = 0.5 * sample_rate

    if f_max > nyquist:
        raise ValueError(
            f"f_max={f_max} exceeds the Nyquist frequency "
            f"{nyquist}."
        )

    # Target frequency grid, exactly as in the attached function
    target_frequencies = np.linspace(
        f_min,
        f_max,
        f_points,
    )

    # Target frequency spacing
    target_spacing = (
        f_max - f_min
    ) / (f_points - 1)

    # Required zero-padded FFT length
    required_n_fft = int(
        np.ceil(
            1.0
            / (dt * target_spacing)
        )
    )

    n_fft = max(len(t), required_n_fft)

    # Positive-frequency FFT grid
    frequencies = np.fft.rfftfreq(
        n_fft,
        d=dt,
    )

    # Same unnormalized magnitude used by plot_fourier_space()
    spectrum = np.abs(
        np.fft.rfft(x, n=n_fft)
    )

    # Interpolate native FFT bins onto target_frequencies
    interpolated_spectrum = np.interp(
        target_frequencies,
        frequencies,
        spectrum,
    )

    return target_frequencies, interpolated_spectrum, n_fft


# ---------------------------------------------------------------------
# Time-domain sampling
# ---------------------------------------------------------------------

dt = 25
N = 6912

t = np.arange(N) * dt
model = time_model(t)


# ---------------------------------------------------------------------
# Calculate spectrum from 0.003 to 0.004 Hz
# ---------------------------------------------------------------------

f_min = 0.003
f_max = 0.004

# Use a moderate value to avoid excessive zero-padding.
# Increase this if you need a smoother interpolated curve.
f_points = 200

frequencies, model_spectrum, n_fft = (
    interpolated_fft_spectrum(
        x=model,
        t=t,
        f_min=f_min,
        f_max=f_max,
        f_points=f_points,
    )
)


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

print(f"Time samples: {len(t)}")
print(f"Time step: {dt} seconds")
print(f"Total time: {t[-1] - t[0]:.2f} seconds")
print(f"FFT length after zero-padding: {n_fft}")
print(f"Native FFT spacing: {1 / (n_fft * dt):.6g} Hz")

plt.figure(figsize=(10, 5))

plt.plot(
    frequencies,
    model_spectrum,
    color="red",
    linewidth=0.8,
    label="Model",
)

# Add the signal frequencies as vertical dashed lines
for i, f in enumerate(fs):
    if f_min <= f <= f_max:
        plt.axvline(
            f,
            color="black",
            linestyle="--",
            linewidth=0.5,
            label="Signals" if i == 0 else None,
        )

plt.xlim(f_min, f_max)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.title("Fourier Spectrum: 0.003–0.004 Hz")
plt.legend()
plt.tight_layout()
plt.show()