import os

# Force math backends to single-thread to prevent deadlocking inside worker processes.
# MUST be set before importing numpy!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import jax
import jax.numpy as jnp
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from tqdm import tqdm

class DataFetcher:
    def __init__(self):
        self.client = Client('IRIS')

    def get_observed_data(self, network, station, channel, location, stream_index, start_time, end_time):
        inventory = self.client.get_stations(network=network, station=station, location=location, 
                                             channel=channel, starttime=start_time, endtime=end_time, level='response')
        
        stream = self.client.get_waveforms(network=network, station=station, location=location, 
                                           channel=channel, starttime=start_time, endtime=end_time)
        
        trace = stream[stream_index]
        trace.detrend('constant')
        trace.remove_response(inventory=inventory, output="ACC")
        
        trace.decimate(5, no_filter=False)
        trace.decimate(5, no_filter=False)

        trace.filter('bandpass', freqmin=0.0002, freqmax=0.010)

        delta = trace.stats.delta 
        N = len(trace)
        t = np.arange(N) * delta
        
        d = np.array(trace.data)

        return t, d

class BATS:
    def __init__(self, t, d):
        self.t = jnp.array(t)
        self.d = jnp.array(d)
        # Assuming a scale factor to normalize data and avoid float overflow during matrix ops
        self.scale = jnp.max(jnp.abs(self.d)) if jnp.max(jnp.abs(self.d)) > 0 else 1.0

    def get_model_log_probability(self, model_frequencies, model_decay_rates):
        """Helper to compute log probability for the Hessian wrapper."""
        omegas = jnp.array(model_frequencies) * 2.0 * jnp.pi
        alphas = jnp.array(model_decay_rates)
        
        t = self.t
        d = self.d / self.scale
        r = len(omegas)
        m = 2 * r
        N = len(d)

        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])

        G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
        gram = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
        eigenvalues = jnp.maximum(eigenvalues, 1e-12)

        H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
        h = H @ d

        sum_sq_data = jnp.sum(d ** 2)
        sum_sq_proj = jnp.sum(h ** 2)
        ratio = sum_sq_proj / sum_sq_data
        
        return 0.5 * (m - N) * jnp.log(1.0 - ratio)

    def get_model_statistics(self, model_frequencies, model_decay_rates):
        # Convert frequency to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)

        # Initialize time t and data d
        t = self.t
        d = self.d / self.scale

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
        estimated_noise_variance = (1 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2))
        SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)

        # Create wrapper function for finding the log probability from the position vector
        def log_probability_wrapper(q):
            mod_freqs = q[:r]
            mod_decays = q[r:]
            return self.get_model_log_probability(mod_freqs, mod_decays)
        
        # Initialize jax functions for the Hessian matrix of the probability distribution
        get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))

        # Find the b matrix (Bretthorst Eq. 4.11)
        b = (-m / 2) * get_log_probability_hessian(jnp.concatenate([jnp.array(model_frequencies), jnp.array(model_decay_rates)]))

        eigenvalues_b, eigenvectors_b = jnp.linalg.eigh(b)
        eigenvalues_b = jnp.maximum(eigenvalues_b, 1e-8)

        # Find the parameter uncertainties (Bretthorst Eq. 4.13)
        parameter_uncertainties = jnp.sqrt(estimated_noise_variance * jnp.sum((eigenvectors_b**2) / eigenvalues_b, axis=1))

        return log_probability, SNR, estimated_noise_variance, parameter_uncertainties


def plot_sigma_regions(results_df):
    """Plots the frequency-decay rate space with 1, 2, and 3 sigma regions."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for _, row in results_df.iterrows():
        # Use the microHertz columns for plotting
        f_val = row['Frequency (uHz)']
        k_val = row['Decay Rate']
        f_unc = row['f_unc (uHz)']
        k_unc = row['k_unc']
        
        # Center points
        ax.scatter(f_val, k_val, color='black', s=15, zorder=5)
        
        # 1, 2, 3 Sigma Ellipses
        for num_sig, alpha in zip([1, 2, 3], [0.4, 0.2, 0.1]):
            ellipse = Ellipse((f_val, k_val), width=f_unc * 2 * num_sig, height=k_unc * 2 * num_sig, 
                              color='blue', alpha=alpha, zorder=4-num_sig)
            ax.add_patch(ellipse)

    ax.set_xlabel("Frequency (µHz)")
    ax.set_ylabel("Decay Rate (1/s)")
    ax.set_title("Mode Frequencies vs Decay Rates with Uncertainty Bounds")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # ---------------------------------------------------------
    # 1. USER VARIABLES
    # ---------------------------------------------------------
    # TOGGLE: True to calculate SNR/download data, False to just plot CSV values
    CALCULATE_SNR = True 

    NETWORK = "IU"
    STATION = "KIP"
    LOCATION = "00"
    CHANNEL = "LHZ"
    STREAM_INDEX = 0
    
    # Time frame covering background + event
    START_TIME = UTCDateTime('2025-07-31T06:24:50')
    END_TIME = UTCDateTime('2025-08-11T05:24:50')
    
    # Event start (used to crop out the initial noise buffer)
    EVENT_START_TIME = UTCDateTime('2025-07-31T11:24:50')
    
    # Path to CSV
    CSV_PATH = "data/earth_normal_modes_table.csv" 
    
    # ---------------------------------------------------------
    # 2. LOAD CSV
    # ---------------------------------------------------------
    print("Importing csv...")
    df = pd.read_csv(CSV_PATH)
    
    # Initialize SNR column for saving later
    if 'SNR' not in df.columns:
        df['SNR'] = np.nan
        
    print("csv imported!")
    results = []

    # ---------------------------------------------------------
    # 3. CONDITIONAL PROCESSING (SNR Calculation vs Plot Only)
    # ---------------------------------------------------------
    if CALCULATE_SNR:
        print("Collecting data...")
        fetcher = DataFetcher()
        master_trace = fetcher.get_observed_data(NETWORK, STATION, CHANNEL, LOCATION, STREAM_INDEX, START_TIME, END_TIME)
        print("Data collected!")

        print("Processing modes...")
        # Wrap tqdm directly around the iterator so it updates automatically
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            f_uhz = row['f_obs']
            k_val = row['k_obs']
            f_unc = row['f_unc']
            k_unc = row['k_unc']
            
            # Skip rows missing necessary data
            if pd.isna(f_uhz) or pd.isna(k_val) or pd.isna(f_unc) or pd.isna(k_unc):
                continue
                
            # Convert microHertz to Hertz, bandwidth is 3x the frequency uncertainty
            f_hz = f_uhz / 1e6
            bw_hz = (5.0 * f_unc) / 1e6
            
            # Bandpass filter around specific frequency
            tr_filtered = master_trace.copy()
            tr_filtered.filter('bandpass', freqmin=f_hz - (bw_hz/2.0), freqmax=f_hz + (bw_hz/2.0))
            
            # Calculate decay time to 0.1% amplitude
            t_decay_seconds = -np.log(0.001) / k_val
            
            # Crop data from Event Start up to the decay time
            tr_cropped = tr_filtered.slice(starttime=EVENT_START_TIME, endtime=EVENT_START_TIME + t_decay_seconds)
            
            # Skip if cropped trace is empty
            if len(tr_cropped) == 0:
                continue
                
            t_arr = np.arange(len(tr_cropped)) * tr_cropped.stats.delta
            d_arr = tr_cropped.data
            
            # --- INTERACTIVE PLOT ---
            mode_name = row.get('mode', f"Mode_{index}")
            plt.figure(figsize=(10, 4))
            plt.plot(t_arr, d_arr, color='black', linewidth=0.8)
            plt.title(f"Filtered Trace: {mode_name} | Freq: {f_uhz} µHz | BW: {f_unc * 3:.2f} µHz")
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show() # Halts execution until manually closed
            
            # Initialize BATS and get stats
            bats_instance = BATS(t_arr, d_arr)
            log_prob, snr, noise_var, _ = bats_instance.get_model_statistics([f_hz], [k_val])
            
            # Save SNR back to the dataframe for exporting
            df.at[index, 'SNR'] = float(snr)
            
            results.append({
                'Mode': mode_name,
                'Frequency (uHz)': f_uhz,
                'Frequency (Hz)': f_hz,
                'Decay Rate': k_val,
                'SNR': float(snr),
                'f_unc (uHz)': float(f_unc),
                'f_unc (Hz)': float(f_unc) / 1e6,
                'k_unc': float(k_unc)
            })
            
        # Export the updated CSV
        base_name, ext = os.path.splitext(CSV_PATH)
        out_csv_path = f"{base_name}_kamchatka_{STATION}{ext}"
        df.to_csv(out_csv_path, index=False)
        print(f"\nSaved updated CSV with SNR to: {out_csv_path}")
            
    else:
        print("CALCULATE_SNR is False. Directly processing CSV for plotting...")
        for index, row in df.iterrows():
            f_uhz = row['f_obs']
            k_val = row['k_obs']
            f_unc = row['f_unc']
            k_unc = row['k_unc']

            # Skip rows missing necessary data
            if pd.isna(f_uhz) or pd.isna(k_val) or pd.isna(f_unc) or pd.isna(k_unc):
                continue
                
            results.append({
                'Mode': row.get('mode', f"Mode_{index}"),
                'Frequency (uHz)': f_uhz,
                'Decay Rate': k_val,
                'SNR': row.get('SNR', 0.0), # Pull SNR from CSV if it exists, otherwise 0
                'f_unc (uHz)': float(f_unc),
                'k_unc': float(k_unc)
            })

    # ---------------------------------------------------------
    # 4. FORMAT RESULTS AND PLOT
    # ---------------------------------------------------------
    if len(results) == 0:
        print("No valid modes found to plot.")
    else:
        results_df = pd.DataFrame(results)
        
        # Only sort by SNR if we actually calculated it
        if CALCULATE_SNR:
            results_df = results_df.sort_values(by='SNR', ascending=False).reset_index(drop=True)
            print("--- Sorted Results by SNR ---")
            print(results_df[['Mode', 'Frequency (uHz)', 'Decay Rate', 'SNR', 'f_unc (uHz)', 'k_unc']])
        else:
            print(f"--- Loaded {len(results_df)} valid modes from CSV for plotting ---")
        
        plot_sigma_regions(results_df)