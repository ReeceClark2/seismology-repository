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

    t = t[144:]
    d = d[144:]

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

# t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
# d = np.sin(2 * np.pi * 0.0011 * t) + np.sin(2 * np.pi * 0.0007 * t) + np.sin(2 * np.pi * 0.0009 * t)

model_frequencies = [
    0.0008140924, 
    0.0009423955, 
    0.0006483128, 
    0.0008413425, 
    0.0010398962, 
    0.0009442520, 
    0.0006453423, 
    0.0008385538, 
    0.0011061142, 
    0.0009368644, 
    0.0006799325, 
    0.0006436609, 
    0.0006797992, 
    0.0011744637, 
    0.0009408306, 
    0.0006761783
]

model_decay_rates = [
    0.000000524483, 
    0.000005551508, 
    0.000007402194, 
    0.000008990732, 
    0.000010137166, 
    0.000005014780, 
    0.000006072663, 
    0.000008203695, 
    0.000009240335, 
    0.000008495631, 
    0.000009482316, 
    0.000004001539, 
    0.000015529109, 
    0.000013982052, 
    0.000005912870, 
   -0.000012458274
]

log_probability, SNR, estimated_noise_variance, parameter_uncertainties, h, H = get_model_statistics(t, d, model_frequencies, model_decay_rates)

model = np.zeros(len(t))
for ind, _ in enumerate(h):
    model += h[ind] * H[ind]

plt.figure(figsize=(10, 6))

# Plot the three series
plt.plot(t, d, label='Data ($d$)', color='black', alpha=0.25, linewidth=1)
plt.plot(t, model, label='Model', color='#d62728', linewidth=1.5)
plt.plot(t, d - model, label='Residual ($d - model$)', color="#082e49", alpha=0.8, linewidth=1)

# Axis labels with LaTeX formatting
plt.xlabel("Time (s)", fontsize=11)
plt.ylabel("Acceleration (m/s$^2$)", fontsize=11)

# Simplistic styling
plt.grid(False) # Ensure no grid
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend without a frame
plt.legend(frameon=False, loc='upper right')

plt.tight_layout()
plt.show()