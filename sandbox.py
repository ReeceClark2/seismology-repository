import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal
import math
from BATS4 import BATS
import jax.numpy as jnp
import jax.scipy.special as jsp
import jax

jax.config.update("jax_enable_x64", True)


def get_synthetic_data(filename, minimum_frequency=None, maximum_frequency=None):
    data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')

    # Column 0 is time in seconds, Column 1 is intensity
    t = data.iloc[:, 0].values
    intensities = data.iloc[:, 1].values
    
    # Calculate actual delta from the file data
    delta = np.mean(np.diff(t)) 
    fs = 1.0 / delta

    nyquist = 0.5 * fs
    low = minimum_frequency / nyquist
    high = maximum_frequency / nyquist

    order = 4
    b, a = signal.butter(order, [low, high], btype='band')

    # Detrending prevents edge offsets from blowing up the filter
    detrended = signal.detrend(intensities)

    d = signal.filtfilt(b, a, detrended)

    # t = t[144:]
    # d = d[144:]

    return t, d

def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')

    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

    trace = stream[stream_index]

    trace.detrend('constant')

    trace.remove_response(inventory=inventory, output="ACC")

    if minimum_frequency and maximum_frequency:
        trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
        
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    delta = trace.stats.delta 
    N = len(trace)
    t = np.arange(N) * delta
    
    d = np.array(trace.data)

    t = t[144:]
    d = d[144:]

    return t, d


def get_model_log_probability(t, d, model_frequencies, model_decay_rates):
    # Convert frequency to angular frequency
    omegas = jnp.array(model_frequencies) * 2 * jnp.pi
    alphas = jnp.array(model_decay_rates)

    # Find the number of functions r, model rank m, and total number of data points N
    r = len(omegas)
    m = 2 * r
    N = len(d)

    # Construct the sinusoid input and decay rates
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])
    
    # Construct the cosine and sine components
    G_cos = jnp.cos(arg) * decay
    G_sin = jnp.sin(arg) * decay
    
    # Create the model function vector G and Gram matrix g (Bretthorst Page 32)
    G = jnp.vstack((G_cos, G_sin))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    # Find the orthonormal functions H (Bretthorst Eq. 3.6)
    H = scaled_eigenvectors.T @ G

    # Find the orthonormal function amplitudes h (Bretthorst Eq. 3.13)
    h = H @ d

    # Find the mean-square of the data (Bretthorst Page 17) and mean of the square projection (Bretthorst Eq. 3.15)
    mean_square_data = (1 / N) * jnp.sum(d ** 2)
    mean_square_projection = (1 / m) * jnp.sum(h ** 2)

    # Find probability (Bretthorst Eq. 3.17).
    ratio = (m * mean_square_projection) / (N * mean_square_data)
    log_probability = 0.5 * (m - N) * jnp.log10(1 - ratio)

    return log_probability



def get_model_statistics(t, d, model_frequencies, model_decay_rates):
    # Convert frequency to angular frequency
    omegas = jnp.array(model_frequencies) * 2 * jnp.pi
    alphas = jnp.array(model_decay_rates)

    # Find the number of functions r, model rank m, and total number of data points N
    r = len(omegas)
    m = 2 * r
    N = len(d)

    # Construct the sinusoid input and decay rates
    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])
    
    # Construct the cosine and sine components
    G_cos = jnp.cos(arg) * decay
    G_sin = jnp.sin(arg) * decay
    
    # Create the model function vector G and Gram matrix g (Bretthorst Page 32)
    G = jnp.vstack((G_cos, G_sin))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    # Find the orthonormal functions H (Bretthorst Eq. 3.6)
    H = scaled_eigenvectors.T @ G

    # Find the orthonormal function amplitudes h (Bretthorst Eq. 3.13)
    h = H @ d

    # Find the mean-square of the data (Bretthorst Page 17) and mean of the square projection (Bretthorst Eq. 3.15)
    mean_square_data = (1 / N) * jnp.sum(d ** 2)
    mean_square_projection = (1 / m) * jnp.sum(h ** 2)

    # Find probability (Bretthorst Eq. 3.17).
    ratio = (m * mean_square_projection) / (N * mean_square_data)
    log_probability = 0.5 * (m - N) * jnp.log10(1 - ratio)

    # Find the estimated noise variance (Bretthorst Eq. 4.7) and SNR (Bretthorst Eq. 4.8)
    estimated_noise_variance = np.abs((1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2)))
    print(mean_square_projection, estimated_noise_variance, m, N)
    SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)
    print(SNR)

    # Create wrapper function for finding the log probability from the position vector
    def log_probability_wrapper(q):
        model_frequencies = q[:r]
        model_decay_rates = q[r:]

        log_probability = get_model_log_probability(t, d, model_frequencies, model_decay_rates)

        return log_probability
    
    # Initialize jax functions for the Hessian matrix of the probability distribution
    get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))

    # Find the b matrix (Bretthorst Eq. 4.11)
    b = (-m / 2) * get_log_probability_hessian(np.concatenate([model_frequencies, model_decay_rates]))

    eigenvalues, eigenvectors = jnp.linalg.eigh(b)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    # Find the parameter uncertainties (Bretthorst Eq. 4.13)
    parameter_uncertainties = jnp.sqrt(estimated_noise_variance * jnp.sum((eigenvectors**2) / eigenvalues, axis=1))

    return log_probability, SNR, estimated_noise_variance, parameter_uncertainties, h, H


def get_fft(t, d, minimum_frequency=None, maximum_frequency=None):
        # Find the sample rate
        delta = np.mean(np.diff(t)) 

        # Apply a window function
        taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
        d = d * taper
        
        # Find the FFT
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 4)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d, n=nfft) * delta

        # Mask the FFT to the specified frequency range
        mask = np.ones_like(frequencies, dtype=bool)

        if minimum_frequency is not None:
            mask &= (frequencies >= minimum_frequency) 

        if maximum_frequency is not None:
            mask &= (frequencies <= maximum_frequency)

        # Return the frequencies and powers with the mask
        return frequencies[mask], power[mask]


minimum_frequency = 0.0005     # Minimum frequency (Hz)
maximum_frequency = 0.0012     # Maximum frequency (Hz)

network = "IU"                  # Network
station = "KIP"                 # Station
channel = "LHZ"                 # Channel
location = "00"                 # Location

stream_index = 0                                    # Stream index
start_time = UTCDateTime('2025-07-31T06:24:50')     # Start time
end_time = UTCDateTime('2025-08-11T05:24:50')       # End time

file_path = f"timeseries_Russia/{network}_{station}_TS.ascii" 

t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
# t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
# d = np.sin(2 * np.pi * 0.0011 * t) + np.sin(2 * np.pi * 0.0007 * t) + np.sin(2 * np.pi * 0.0009 * t)

model_frequencies = [
    0.0008146543377901, 0.0009427651393473, 0.0009447533952135, 0.0006426282150699, 0.0006459406497157, 
    0.0008395379785234, 0.0008423188720411, 0.0010396741910682, 0.0010364924080598, 0.0006460364750220, 
    0.0006508961651642, 0.0006846456387499, 0.0006843798713934, 0.0006755903116709
]

model_decay_rates = [
    0.0000005084484921, 0.0000041179925748, 0.0000468724556667, 0.0000054184184974, 0.0000087759101338, 
    0.0000043545628236, 0.0000082438151346, 0.0000121257909676, 0.0000142483803974, 0.0000214094131967, 
    0.0000152861593340, 0.0000075731028185, 0.0000630123961435, 0.0000300282154905
]

log_probability, SNR, estimated_noise_variance, parameter_uncertainties, h, H = get_model_statistics(t, d, model_frequencies, model_decay_rates)

model = np.zeros(len(t))
for ind, _ in enumerate(h):
    model += h[ind] * H[ind]

f, p = get_fft(t, d, minimum_frequency, maximum_frequency)
f1, p1 = get_fft(t, d - model, minimum_frequency, maximum_frequency)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left Plot: Time Series ---
ax1.plot(t, d, label='Data ($d$)', color='black', alpha=0.25, linewidth=1)
ax1.plot(t, model, label='Model', color="#082e49", linewidth=1.5)
ax1.plot(t, d - model, label='Residual ($d - model$)', color='#d62728', alpha=0.8, linewidth=1)

ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Acceleration (m/s$^2$)", fontsize=11)
ax1.legend(frameon=False, loc='upper right')

# --- Right Plot: FFT ---
ax2.plot(f, abs(p), color='black', linewidth=1, label='Data Power Spectrum')
ax2.plot(f1, abs(p1), color='#d62728', linewidth=1, label='Residual Power Spectrum')

# Add vertical lines for model frequencies
for i, freq in enumerate(model_frequencies):
    label = 'Model Frequencies' if i == 0 else None # Label only the first one for the legend
    ax2.axvline(x=freq, color='#d62728', linestyle='--', alpha=0.5, linewidth=1, label=label)

ax2.set_xlabel("Frequency (Hz)", fontsize=11)
ax2.set_ylabel("Power", fontsize=11)
ax2.legend(frameon=False, loc='upper right')

# --- Shared Simplistic Styling ---
for ax in [ax1, ax2]:
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.show()