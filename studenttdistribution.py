import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt


rng = np.random.default_rng()       # Initialize numpy random number generator

periods = 10                        # Number of periods for sinusoid to repeat
N = 1000                            # Number of samples to create

x = np.linspace(0, periods, N)              # Create evenly sampled times to collect data at
e = rng.uniform(low=-2.6, high=2.6, size=N) # Create white noise in specified range
y = np.sin(2 * np.pi * x) + e               # Create sinusoid with noise

omega = np.linspace(0.5, 10, 1000)                          # Create frequencies to sample (Bretthorst specifies dimensionless frequencies which are accounted for later)
inner_term = y * np.exp(1j * omega[:, np.newaxis] * x)         
summed_signal = np.sum(inner_term, axis=1)                  # Discrete Fourier transform
C = (1 / len(x)) * np.abs(summed_signal)                    # Schuster periodogram (Bretthorst Eq. 1.1)
d = (1 / N) * np.sum(y ** 2)                                # Observed mean-square data value

probability = (1 - ((2 * C) / (N * d))) ** ((2 - N) / 2)    # Probability calculation with no known variance (Bretthorst Eq. 2.8)
probability -= min(probability)
probability /= max(probability)

variance = 0.05                                     # Introduce a known variance (the variance should be 0 for a perfect sinusoid, but this creates a computational issue when displaying a Dirac delta)
probability2 = np.exp(C / variance ** 2)            # Probability calculation with known variance (Bretthorst Eq. 2.7)         
total_integral = trapezoid(probability2, x=omega)   # Use scipy's integrate function

power = 2 * (variance + C) * (np.exp(C / variance ** 2) / (total_integral))
power -= min(power)
power /= max(power)                         # Normalize power to 1 (this defeats the purpose of calculating power but makes it easier to compare to FFT)

dt = x[1] - x[0]                            # Find sampling delta
yf = np.fft.fft(y)                          # Carry out numpy FFT for intensities
xf = np.fft.fftfreq(N, d=dt)                # Carry out numpy FFT for frequencies

fft_mag = np.abs(yf)
fft_mag -= min(fft_mag)
fft_mag_norm = fft_mag / np.max(fft_mag)    # Normalize FFT to 1

# Plotting functions below

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

ax1.plot(x, np.sin(2 * np.pi * x) + e, label="Sinusoid + Noise", color="darksalmon")
ax1.plot(x, np.sin(2 * np.pi * x), label="Sinusoid", color="gray")
ax1.set_title("Single Sinusoid with Noise")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Intensity")
ax1.set_xlim(min(x), max(x))
ax1.legend()

ax2.plot(omega / (2 * np.pi), power, label="Power Spectral Density", color="sandybrown", alpha=0.75)
ax2.plot(omega / (2 * np.pi), probability, label="Probability Distribution", color="cornflowerblue", alpha=0.75)
pos_mask = xf >= 0
ax2.plot(xf[pos_mask], fft_mag_norm[pos_mask], label="Basic FFT", color="seagreen", alpha=0.75)
ax2.set_title("Fourier Space")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Power")
ax2.set_xlim(0.4, 1.6)
ax2.legend()

plt.tight_layout()
plt.show()
