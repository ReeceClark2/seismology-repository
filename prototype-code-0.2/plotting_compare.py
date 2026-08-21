import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

import jax
import jax.numpy as jnp

from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client


# ==========================================
# Data Acquisition Functions
# ==========================================
def get_synthetic_data(filename, min_f=None, max_f=None):
    data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')
    t = data.iloc[:, 0].values
    d = data.iloc[:, 1].values

    delta = np.mean(np.diff(t))
    fs = 1.0 / delta
    nyquist = 0.5 * fs
    low = min_f / nyquist
    high = max_f / nyquist

    b, a = signal.butter(4, [low, high], btype='band')
    detrended = signal.detrend(d)
    d = signal.filtfilt(b, a, detrended)

    return t[144:], d[144:]


def get_observed_data(network, station, channel, location, stream_index,
                      start_time, end_time, min_f, max_f):
    client = Client('IRIS')
    inventory = client.get_stations(network=network, station=station,
                                    location=location, channel=channel,
                                    starttime=start_time, endtime=end_time,
                                    level='response')
    stream = client.get_waveforms(network=network, station=station,
                                  location=location, channel=channel,
                                  starttime=start_time, endtime=end_time)
    trace = stream[stream_index]
    trace.detrend('constant')
    trace.remove_response(inventory=inventory, output="ACC")
    
    if min_f and max_f:
        trace.filter('bandpass', freqmin=min_f, freqmax=max_f)
        
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    delta = trace.stats.delta
    N = len(trace)
    t = np.arange(N) * delta
    d = np.array(trace.data)
    
    return t[144:], d[144:]


# ==========================================
# Bayesian Model Statistics (JAX)
# ==========================================
def get_model_log_probability(t, d, model_frequencies, model_decay_rates):
    omegas = jnp.array(model_frequencies) * 2 * jnp.pi
    alphas = jnp.array(model_decay_rates)
    r = len(omegas)
    m = 2 * r
    N = len(d)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])
    
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)
    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    H = scaled_eigenvectors.T @ G
    h = H @ d

    mean_square_data = (1 / N) * jnp.sum(d ** 2)
    mean_square_projection = (1 / m) * jnp.sum(h ** 2)
    ratio = (m * mean_square_projection) / (N * mean_square_data)
    
    return 0.5 * (m - N) * jnp.log10(1 - ratio)


def get_model_statistics(t, d, model_frequencies, model_decay_rates):
    omegas = jnp.array(model_frequencies) * 2 * jnp.pi
    alphas = jnp.array(model_decay_rates)
    r = len(omegas)
    m = 2 * r
    N = len(d)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])
    
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)
    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    H = scaled_eigenvectors.T @ G
    h = H @ d

    mean_square_projection = (1 / m) * jnp.sum(h ** 2)
    estimated_noise_variance = jnp.abs((1 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2)))
    SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** 0.5
    
    # Hessian for uncertainties
    def log_probability_wrapper(q):
        return get_model_log_probability(t, d, q[:r], q[r:])
    
    get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))
    params = jnp.concatenate([jnp.array(model_frequencies), jnp.array(model_decay_rates)])
    b = (-m / 2) * get_log_probability_hessian(params)

    b_eigenvalues, b_eigenvectors = jnp.linalg.eigh(b)
    b_eigenvalues = jnp.maximum(b_eigenvalues, 1e-8)

    parameter_uncertainties = jnp.sqrt(estimated_noise_variance * jnp.sum((b_eigenvectors**2) / b_eigenvalues, axis=1))
    log_prob = log_probability_wrapper(params)

    return log_prob, SNR, estimated_noise_variance, parameter_uncertainties, h, H


# ==========================================
# Processing & Plotting Logic
# ==========================================
def process_model_csv(csv_path, t, d):
    """Loads a CSV, calculates model statistics, and returns the time-domain residual."""
    print(f"Processing {csv_path}...")
    df = pd.read_csv(csv_path, skiprows=1)
    df.columns = df.columns.str.strip()
    
    freqs = df['frequency'].dropna().values
    decays = df['decay_rate'].dropna().values
    
    _, _, _, _, h, H = get_model_statistics(t, d, freqs, decays)
    
    model = np.array(h @ H)
    return d - model


def plot_noise_profiles(res_init, res_nuts, sample_rate, pad_factor=5):
    """Computes and overlays the FFT strictly for the two model residuals."""
    window = signal.get_window(('kaiser', 2. * np.pi), len(res_init))
    res_init *= window

    window = signal.get_window(('kaiser', 2. * np.pi), len(res_nuts))
    res_nuts *= window

    N_original = len(res_init)
    N_padded = N_original * pad_factor
    
    xf = fftfreq(N_padded, 1 / sample_rate)
    positive_freqs = xf[:N_padded//2]
    
    def get_magnitude(data):
        yf = fft(data, n=N_padded)
        return np.abs(yf[:N_padded//2]) * (2.0 / N_original)

    mag_init = get_magnitude(res_init)
    mag_nuts = get_magnitude(res_nuts)

    plt.figure(figsize=(12, 6))
    
    plt.plot(positive_freqs, mag_init, label='Initial Conditions (40_model.csv)', color='orange', alpha=0.8, linewidth=1.5)
    plt.plot(positive_freqs, mag_nuts, label='NUTS Optimized (40_model_long.csv)', color='purple', alpha=0.8, linewidth=1.5)
    
    # Restrict x-axis to the bandpass filter range
    plt.xlim(0.00025, 0.00160)
    
    plt.title(f"Noise Profile Comparison in Fourier Space (Zero-padded {pad_factor}x)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude of Residuals")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    # Parameters
    min_f = 0.00025
    max_f = 0.00160
    network, station, channel, location = "IU", "KIP", "LHZ", "00"
    stream_index = 0
    start_time = UTCDateTime('2025-07-31T06:24:50')
    end_time = UTCDateTime('2025-08-11T05:24:50')
    data_source = "observed"

    # 1. Fetch Time Series Data
    print(f"Fetching {data_source} data...")
    if data_source == "observed":
        t, d = get_observed_data(network, station, channel, location,
                                 stream_index, start_time, end_time,
                                 min_f, max_f)
    else:
        file_path = f"../timeseries-kamchatka/{network}_{station}_TS.ascii"
        t, d = get_synthetic_data(file_path, min_f, max_f)

    fs = 1.0 / (t[1] - t[0])
    print(f"Data loaded. N={len(d)}, Sampling rate: {fs:.4f} Hz\n")

    # 2. Process both models to get their residuals
    residual_init = process_model_csv('40_model.csv', t, d)
    residual_nuts = process_model_csv('40_model_long.csv', t, d)

    # 3. Plot the overlaid noise profiles in Fourier Space
    print("\nGenerating Fourier Space Plot...")
    plot_noise_profiles(residual_init, residual_nuts, fs)