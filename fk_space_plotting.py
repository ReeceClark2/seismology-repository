import jax
import jax.numpy as jnp
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from obspy.clients.fdsn import Client
from scipy.ndimage import median_filter
from obspy.core import UTCDateTime

# ==========================================
# 1. Configuration & File Paths
# ==========================================
DATASET1_CSV_PATH = "C:/Users/starb/Downloads/dracula_output_3_4_KIP/N032_signals.csv"
DATASET2_CSV_PATH = "C:/Users/starb/Downloads/dracula_output_20260911_144101_833340/N022_signals.csv" # Update this path

MAX_DECAY_RATE = 8e-2  

network = "IU"
station = "KIP"
location = "00"
channel = "LHZ"
stream_index = 0
start_time = UTCDateTime("2025-07-29T23:24:50")
end_time = UTCDateTime("2025-08-06T05:24:50")
min_f = 0.0030
max_f = 0.0040

# ==========================================
# 2. Amplitude Extraction (JAX Bretthorst)
# ==========================================
@jax.jit
def get_physical_amplitudes(t: jax.Array, d: jax.Array, fs: jax.Array, ks: jax.Array) -> jax.Array:
    omegas = fs * 2.0 * jnp.pi
    r = omegas.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    transform = eigenvectors / jnp.sqrt(eigenvalues)
    H = transform.T @ G
    h = H @ d
    B = transform @ h

    B_cos = B[:r]
    B_sin = B[r:]
    return jnp.sqrt(B_cos**2 + B_sin**2)

# ==========================================
# 3. Data Loading & Cleaning
# ==========================================
def load_and_filter_data(csv_path, max_decay):
    df = pd.read_csv(csv_path).dropna(
        subset=[
            'frequencies',
            'frequency_uncertainties',
            'decay_rates',
            'decay_rate_uncertainties',
        ]
    ).copy()
    return df[df['decay_rates'] <= max_decay]

df1 = load_and_filter_data(DATASET1_CSV_PATH, MAX_DECAY_RATE)
df2 = load_and_filter_data(DATASET2_CSV_PATH, MAX_DECAY_RATE)

# ==========================================
# 4. Time Series & Amplitude Calculation
# ==========================================
def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, min_f, max_f):
    client = Client("IRIS")
    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level="response")
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)
    
    trace = stream[stream_index]
    trace.detrend("constant")
    trace.remove_response(inventory=inventory, output="ACC")
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.filter("bandpass", freqmin=min_f, freqmax=max_f, corners=16, zerophase=True)

    delta = float(trace.stats.delta)
    t = np.arange(len(trace), dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)
    return t, d

print("Downloading and preparing observed data...")
t_arr, d_arr = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, min_f, max_f)

amp1 = np.array(get_physical_amplitudes(t_arr, d_arr, jnp.array(df1['frequencies'].values), jnp.array(df1['decay_rates'].values)))
amp2 = np.array(get_physical_amplitudes(t_arr, d_arr, jnp.array(df2['frequencies'].values), jnp.array(df2['decay_rates'].values)))

# ==========================================
# 5. Fourier Space Lorentzian Summation
# ==========================================
f_grid = np.linspace(2.9e-3, 4.1e-3, 5000)
gammas1 = df1['decay_rates'].values / (2.0 * np.pi)
gammas2 = df2['decay_rates'].values / (2.0 * np.pi)

def sum_weighted_lorentzians(f_eval, f_centers, gammas, amplitudes):
    diff = f_eval[np.newaxis, :] - f_centers[:, np.newaxis]
    denom = diff**2 + gammas[:, np.newaxis] ** 2
    profile = 0.25 * (amplitudes[:, np.newaxis] ** 2) / denom
    return np.sum(profile, axis=0)

spec1 = sum_weighted_lorentzians(f_grid, df1['frequencies'].values, gammas1, amp1)
spec2 = sum_weighted_lorentzians(f_grid, df2['frequencies'].values, gammas2, amp2)

# Baseline removal
window_size = 5000 
spec1_clean = np.maximum(spec1 - median_filter(spec1, size=window_size), 0)
spec2_clean = np.maximum(spec2 - median_filter(spec2, size=window_size), 0)

# Normalization & Residual
spec1_norm = spec1_clean / np.max(spec1_clean)
spec2_norm = spec2_clean / np.max(spec2_clean)
residual = spec2_norm - spec1_norm

# ==========================================
# 6. Plotting
# ==========================================
fig, ax1 = plt.subplots(figsize=(11, 6))

def plot_sigma_ellipses(ax, df, base_color, label):
    for _, row in df.iterrows():
        for num_sigma, alpha_val in zip([2, 1], [0.1, 0.2]):
            ell = Ellipse(xy=(row['frequencies'], row['decay_rates']),
                          width=2 * row['frequency_uncertainties'] * num_sigma,
                          height=2 * row['decay_rate_uncertainties'] * num_sigma,
                          facecolor=base_color, edgecolor='none', alpha=alpha_val)
            ax.add_patch(ell)
    ax.semilogy(df['frequencies'], df['decay_rates'], marker='.', color=base_color, linestyle='None', markersize=4, label=label)

# Plot the data
plot_sigma_ellipses(ax1, df1, 'blue', 'Dataset 1')
plot_sigma_ellipses(ax1, df2, 'red', 'Dataset 2')

# Formatting
ax1.set_title('Frequency-Decay Rate Space')
ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Decay Rate')
ax1.set_ylim(7e-6, 1e-3)

# Deduplicate legend handles
handles, labels = ax1.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax1.legend(by_label.values(), by_label.keys())

plt.tight_layout()
plt.show()