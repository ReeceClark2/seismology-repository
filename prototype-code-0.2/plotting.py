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

    b, a = sp_signal.butter(4, [low, high], btype='band')
    detrended = sp_signal.detrend(d)
    d = sp_signal.filtfilt(b, a, detrended)

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
    
    G_cos = jnp.cos(arg) * decay
    G_sin = jnp.sin(arg) * decay
    
    G = jnp.vstack((G_cos, G_sin))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)
    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    H = scaled_eigenvectors.T @ G
    h = H @ d

    mean_square_data = (1 / N) * jnp.sum(d ** 2)
    mean_square_projection = (1 / m) * jnp.sum(h ** 2)

    ratio = (m * mean_square_projection) / (N * mean_square_data)
    log_probability = 0.5 * (m - N) * jnp.log10(1 - ratio)

    return log_probability


def get_model_statistics(t, d, model_frequencies, model_decay_rates):
    omegas = jnp.array(model_frequencies) * 2 * jnp.pi
    alphas = jnp.array(model_decay_rates)

    r = len(omegas)
    m = 2 * r
    N = len(d)

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])
    
    G_cos = jnp.cos(arg) * decay
    G_sin = jnp.sin(arg) * decay
    
    G = jnp.vstack((G_cos, G_sin))
    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)
    scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

    H = scaled_eigenvectors.T @ G
    h = H @ d

    mean_square_projection = (1 / m) * jnp.sum(h ** 2)
    
    # Calculate Noise Variance & SNR
    print(N, m, jnp.sum(d ** 2), jnp.sum(h ** 2))
    estimated_noise_variance = jnp.abs((1 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2)))
    SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** 0.5
    
    print(f"Mean Square Projection: {mean_square_projection:.4e}")
    print(f"Estimated Noise Var: {estimated_noise_variance:.4e}")
    print(f"SNR: {SNR:.4f}")

    # Hessian for uncertainties
    def log_probability_wrapper(q):
        mf = q[:r]
        mdr = q[r:]
        return get_model_log_probability(t, d, mf, mdr)
    
    get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))
    
    params = jnp.concatenate([jnp.array(model_frequencies), jnp.array(model_decay_rates)])
    b = (-m / 2) * get_log_probability_hessian(params)

    b_eigenvalues, b_eigenvectors = jnp.linalg.eigh(b)
    b_eigenvalues = jnp.maximum(b_eigenvalues, 1e-8)

    parameter_uncertainties = jnp.sqrt(estimated_noise_variance * jnp.sum((b_eigenvectors**2) / b_eigenvalues, axis=1))

    # Calculate log prob directly to return
    log_prob = log_probability_wrapper(params)

    return log_prob, SNR, estimated_noise_variance, parameter_uncertainties, h, H


# ==========================================
# Plotting Functions
# ==========================================
def plot_time_series(t, d, model, residual, single_plot=False):
    """Plots original data, constructed model, and the residual in the time domain."""
    if single_plot:
        # Plot everything on one graph
        plt.figure(figsize=(12, 6))
        plt.plot(t, d, label="Original Observed Data", color='black', linewidth=1, alpha=0.5)
        plt.plot(t, model, label="Reconstructed Model", color='blue', linewidth=1, alpha=0.7)
        plt.plot(t, residual, label="Residual (Data - Model)", color='red', linewidth=1, alpha=0.7)
        
        plt.title("Time Series Overview: Data, Model, and Residual")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
    else:
        # Plot on 3 separate subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        
        axs[0].plot(t, d, color='black', linewidth=1)
        axs[0].set_title("Original Observed Data")
        axs[0].set_ylabel("Amplitude")
        axs[0].grid(True, alpha=0.3)

        axs[1].plot(t, model, color='blue', linewidth=1)
        axs[1].set_title("Reconstructed Model")
        axs[1].set_ylabel("Amplitude")
        axs[1].grid(True, alpha=0.3)

        axs[2].plot(t, residual, color='red', linewidth=1)
        axs[2].set_title("Residual (Data - Model)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Amplitude")
        axs[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def plot_fft_comparisons(signal_data, model_data, residual_data, sample_rate, highlight_freqs, pad_factor=5):
    """Computes and overlays the FFT for data, model, and residual alongside CSV frequencies."""
    window = signal.get_window(('kaiser', 2. * np.pi), len(model_data))
    model_data *= window
    signal_data *= window
    residual_data *= window

    N_original = len(signal_data)
    N_padded = N_original * pad_factor
    
    # Calculate frequency bins
    xf = fftfreq(N_padded, 1 / sample_rate)
    positive_freqs = xf[:N_padded//2]
    
    # Helper to compute normalized FFT magnitude
    def get_magnitude(data):
        yf = fft(data, n=N_padded)
        return np.abs(yf[:N_padded//2]) * (2.0 / N_original)

    mag_data = get_magnitude(signal_data)
    mag_model = get_magnitude(model_data)
    mag_residual = get_magnitude(residual_data)

    plt.figure(figsize=(12, 6))
    
    # Plot Spectra
    plt.plot(positive_freqs, mag_data, label='Original Data', color='black', alpha=0.6, linewidth=1.5)
    plt.plot(positive_freqs, mag_model, label='Model', color='blue', alpha=0.8, linewidth=1.5)
    plt.plot(positive_freqs, mag_residual, label='Residual', color='red', alpha=0.6, linewidth=1.5)
    
    # Plot CSV Frequencies
    for i, freq in enumerate(highlight_freqs):
        plt.axvline(x=freq, color='green', linestyle='--', linewidth=1, alpha=0.8,
                    label='CSV Frequencies' if i == 0 else "")

    if len(positive_freqs) > 0:
        plt.xlim([positive_freqs[1], positive_freqs[-1]])

    plt.xlim(0.00025, 0.00160)
    plt.title(f"FFT Spectrum Comparison (Zero-padded {pad_factor}x)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
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

    file_path = f"../timeseries-kamchatka/{network}_{station}_TS.ascii"
    csv_file_path = '42_model_long.csv' 
    data_source = "observed"
    
    # Plotting Toggle
    use_single_plot_layout = False  # Set to False to get 3 separate graphs

    # 1. Load CSV Parameters (Extract Frequencies and Decay Rates)
    print(f"Loading model parameters from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, skiprows=1)
        df.columns = df.columns.str.strip()
        model_frequencies = df['frequency'].dropna().values
        model_decay_rates = df['decay_rate'].dropna().values
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        exit()

    # 2. Fetch Time Series Data
    print(f"Fetching {data_source} data...")
    if data_source == "observed":
        t, d = get_observed_data(network, station, channel, location,
                                 stream_index, start_time, end_time,
                                 min_f, max_f)
    else:
        t, d = get_synthetic_data(file_path, min_f, max_f)

    fs = 1.0 / (t[1] - t[0])
    print(f"Data loaded. N={len(d)}, Sampling rate: {fs:.4f} Hz")

    # 3. Calculate Model Statistics via JAX
    print("Calculating Bayesian model statistics...")
    log_prob, SNR, est_noise_var, param_uncert, h, H = get_model_statistics(
        t, d, model_frequencies, model_decay_rates
    )

    # 4. Construct the Model & Residual
    model = np.array(h @ H) 
    residual = d - model

    # 5. Plotting Results
    print(f"Generating Time Series Plot (Single Plot Layout: {use_single_plot_layout})...")
    plot_time_series(t, d, model, residual, single_plot=use_single_plot_layout)

    print("Generating Fourier Space Plot...")
    plot_fft_comparisons(d, model, residual, fs, model_frequencies)