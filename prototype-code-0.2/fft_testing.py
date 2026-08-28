import numpy as np
from scipy import signal
from scipy.signal import hilbert # Imported hilbert
import math
import matplotlib.pyplot as plt
from BATS import BATS

def get_fft(t, d, min_f, max_f):
        '''
        Helper function for finding the FFT of BATS's provided time series using a 
        Kaiser window. Modified to splice low frequencies (<1.5MHz) from the full 
        dataset with high frequencies (>1.5MHz) from the first 2/5ths of the dataset.
        '''
        # Define the cutoff frequency 
        cutoff_freq = 1.5 
        
        # 1. Process the full dataset
        window_full = signal.get_window(('kaiser', 2. * np.pi), len(d))
        d_full = d #* window_full

        # Calculate time step (dt) instead of assuming diff is frequency
        dt = np.mean(np.diff(t))
        
        # We define nfft based on the full dataset to maintain exact frequency bins
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 2)
        fs = np.fft.fftfreq(n=nfft, d=dt)
        power_full = np.fft.fft(d_full, n=nfft) * dt

        # 2. Process the truncated dataset (first 2/5)
        idx_2_5 = int(len(d) * 2 / 5)
        d_trunc = d[:idx_2_5]
        window_trunc = signal.get_window(('kaiser', 2. * np.pi), len(d_trunc))
        d_trunc = d_trunc * window_trunc
        
        # Use the exact same nfft so the frequency bins (fs) match perfectly
        power_trunc = np.fft.fft(d_trunc, n=nfft) * dt

        # 3. Splice the two sections together
        # Restored the > cutoff_freq condition
        power_combined = np.where(np.abs(fs) > cutoff_freq, power_trunc, power_full)

        # 4. Mask the final FFT to the specified global frequency range
        mask = np.ones_like(fs, dtype=bool)
        mask &= (fs >= min_f) 
        mask &= (fs <= max_f)

        # Return the fs and powers with the mask
        return np.array(fs[mask]), np.array(power_combined[mask])


rng = np.random.default_rng()

min_f = 1
max_f = 10

n = 100_000

t = np.linspace(1, n, 20 * n)
e = rng.uniform(low=-0.1, high=0.1, size=20 * n)

f1 = 2.5
k1 = 1e-2
a1 = 2

f2 = 5
k2 = 2e-3
a2 = 2

d = (a1 * np.sin(2 * np.pi * f1 * t) * np.e ** (-k1 * t) +
     a2 * np.sin(2 * np.pi * f2 * t) * np.e ** (-k2 * t))

Q1 = np.pi * f1 / k1
Q2 = np.pi * f2 / k2
print(f"Q1: {Q1}, Q2: {Q2}")

# 1. Calculate Fourier Transform (using your custom function)
f, p = get_fft(t, d, min_f, max_f)

# --- Plotting Code Below ---

freq_space = np.linspace(1, 4, 10000)
lorentz = k1 / (k1 ** 2 + (freq_space - f1) ** 2)
plt.plot(freq_space, lorentz)
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Time Series & Hilbert Envelope
ax1.plot(t, d, color='tab:blue', alpha=0.6, label='Original Signal')
ax1.set_title('Time Series & Hilbert Envelope')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()
# Limit the X-axis so you can actually see the waveforms and envelope clearly
# The signal decays heavily by t=30 due to your k1 and k2 values
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Frequency Spectrum (Fourier Transform)
ax2.plot(f, np.abs(p), color='tab:orange')
ax2.set_title('Frequency Spectrum (Fourier Transform)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()