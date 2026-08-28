import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal

import jax
import jax.numpy as jnp

from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

# Enable 64-bit precision for JAX (critical for Hessian calculations)
jax.config.update("jax_enable_x64", True)

# ==========================================
# Data Acquisition Functions
# ==========================================
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
def _projection_quantities(t, d, omegas, alphas):
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d

    mean_square_data = jnp.sum(d ** 2) / N
    mean_square_projection = jnp.sum(h ** 2) / m

    return mean_square_data, mean_square_projection, h, eigenvalues, eigenvectors, m, N


@jax.jit
def get_log_probability(fs, ks, t, d):
    omegas = fs * 2.0 * jnp.pi
    msd, msp, _, _, _, m, N = _projection_quantities(t, d, omegas, ks)
    ratio = (m * msp) / (N * msd)
    return 0.5 * (m - N) * jnp.log10(1.0 - ratio)


def get_model_statistics_and_powers(t, d, fs, ks):
    """Calculates model statistics by sampling a high-density local grid around each frequency."""
    fs_jnp = jnp.asarray(fs, dtype=jnp.float64)
    ks_jnp = jnp.asarray(ks, dtype=jnp.float64)
    omegas = fs_jnp * 2.0 * jnp.pi
    
    msd, msp, h, _, _, m, N = _projection_quantities(t, d, omegas, ks_jnp)

    # Estimate Noise Variance (sigma^2)
    sigma_sq = (1.0 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2))
    sigma = jnp.sqrt(sigma_sq)

    # Calculate Hessian to get b matrix
    r = fs_jnp.shape[0]
    def lp_wrapper(q):
        return get_log_probability(q[:r], q[r:], t, d)

    q0 = jnp.concatenate([fs_jnp, ks_jnp])
    hessian = jax.jit(jax.hessian(lp_wrapper))(q0)
    b_matrix = (-m / 2.0) * hessian
    b_diagonal_sum = jnp.trace(b_matrix)

    powers = []
    window = 1e-6  # Defines the +/- Hz range to sample around the true frequency
    
    for f in fs:
        # Use an odd number of samples (1001) to guarantee the exact center 'f' is hit
        f_space = np.linspace(f - window, f + window, 1001)
        
        # Phase argument: 2D array of (frequencies, times)
        phase_arg = 2 * np.pi * f_space[:, None] * t[None, :]
        
        # Periodogram evaluated over the localized f_space grid
        C = (1 / N) * np.abs(np.sum(d[None, :] * np.exp(1j * phase_arg), axis=1)) ** 2
        
        # Bayesian power profile evaluated over the localized grid
        power_profile = (4 / m) * 
        (sigma_sq + C) * np.sqrt(b_diagonal_sum / (2 * np.pi * sigma)) * 
        np.exp((-b_diagonal_sum * (f - f_space)**2) / (2 * sigma**2))
        
        # Extract the absolute peak power found within this localized space
        peak_power = np.max(power_profile)
        powers.append(peak_power)
        
    return np.array(h), m, N, np.array(powers)

# ==========================================
# Plotting Functions
# ==========================================
def plot_dual_axis_overlay(model_data, powers, highlight_freqs, sample_rate, pad_factor=5):
    """Plots model FFT on left axis and percentage Bayesian powers on right axis."""
    window = signal.get_window(('kaiser', 2. * np.pi), len(model_data))
    model_data *= window

    N_original = len(model_data)
    N_padded = N_original * pad_factor
    
    # Calculate FFT of the constructed model
    xf = fftfreq(N_padded, 1 / sample_rate)
    positive_freqs = xf[:N_padded//2]
    
    yf_model = fft(model_data, n=N_padded)
    mag_model = np.abs(yf_model[:N_padded//2]) * (2.0 / N_original)
    
    # Normalize powers by sum so they act as fractional/percentage values
    fractional_powers = powers / np.sum(powers)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # === Primary Axis (Left): FFT Intensity ===
    color1 = 'blue'
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("FFT Magnitude (Intensity)", color=color1)
    ax1.plot(positive_freqs, mag_model, label='Model FFT', color=color1, alpha=0.7, linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(0.00025, 0.00160)
    ax1.grid(True, alpha=0.3)
    
    # Draw vertical lines for the model frequencies on ax1
    for i, freq in enumerate(highlight_freqs):
        ax1.axvline(x=freq, color='green', linestyle='--', linewidth=1, alpha=0.5,
                    label='Model Frequencies' if i == 0 else "")

    # === Secondary Axis (Right): Fractional Powers ===
    ax2 = ax1.twinx()  
    color2 = 'red'
    ax2.set_ylabel("Bayesian Power (Fraction of Total)", color=color2)
    ax2.scatter(highlight_freqs, fractional_powers, color=color2, edgecolor='black', s=80, 
                zorder=5, label='Fractional Power')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Provide a tiny bit of headroom for the scatter plot points
    ax2.set_ylim(0, np.max(fractional_powers) * 1.1)

    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.title("Dual-Axis: Model FFT Intensity and Fractional Bayesian Power")
    fig.tight_layout()
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
    csv_file_path = '42_model_long.csv' 

    print(f"Loading model parameters from {csv_file_path}...")
    try:
        df = pd.read_csv(csv_file_path, skiprows=1)
        df.columns = df.columns.str.strip()
        model_frequencies = df['frequency'].dropna().values
        model_decay_rates = df['decay_rate'].dropna().values
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        exit()

    print("Fetching observed data...")
    t, d = get_observed_data(network, station, channel, location,
                             stream_index, start_time, end_time,
                             min_f, max_f)

    fs_rate = 1.0 / (t[1] - t[0])
    print(f"Data loaded. Sampling rate: {fs_rate:.4f} Hz")

    print("Calculating projection and expected powers...")
    h, m, N, powers = get_model_statistics_and_powers(t, d, model_frequencies, model_decay_rates)

    # Construct the time-domain model
    fs_jnp = jnp.asarray(model_frequencies, dtype=jnp.float64)
    ks_jnp = jnp.asarray(model_decay_rates, dtype=jnp.float64)
    arg = (fs_jnp * 2.0 * jnp.pi)[:, None] * t[None, :]
    decay = jnp.exp(-ks_jnp[:, None] * t[None, :])
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    
    gram = G @ G.T
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    model = np.array(h @ H) 

    print("Generating Dual-Axis Overlay Plot...")
    plot_dual_axis_overlay(model, powers, model_frequencies, fs_rate)