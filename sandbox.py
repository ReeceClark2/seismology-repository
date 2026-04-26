import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal, stats
import math
import random
from pathlib import Path
import itertools
from matplotlib.ticker import MaxNLocator
from BATS import BATS


def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')

    # Use obspy to retrieve data
    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

    # Run bandpass on retrieved data
    trace = stream[stream_index]

    if minimum_frequency and maximum_frequency:
        trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
    trace.decimate(5, no_filter=False)
    trace.decimate(4, no_filter=False)
    trace.detrend('constant')

    # Create accurate time array assuming evenly sampled data
    delta = trace.stats.delta 
    N = len(trace)
    t = np.arange(N) * delta

    t = t[180:]
    d = np.array(trace)
    d = d[180:]

    return t, d


def compute_fft(t, d, minimum_frequency, maximum_frequency):
    delta = np.mean(np.diff(t)) 

    taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
    d = d * taper
    
    nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 3)
    
    frequencies = np.fft.fftfreq(n=nfft, d=delta)
    power = np.fft.fft(d, n=nfft) * delta

    mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    return frequencies[mask], power[mask]


def compute_probability(t, d, model_frequencies, model_quality_factors):
    # Vectorized computation (Bretthorst)
    m_f = 2 * np.pi * np.array(model_frequencies)
    m_d = m_f / (2 * np.array(model_quality_factors))

    r = len(model_frequencies)
    m = 2 * r
    N = len(d)

    # 1. Populate model function arrays via broadcasting (Bretthorst page 32).
    m_f_col = m_f[:, np.newaxis]
    m_d_col = m_d[:, np.newaxis]

    arg = m_f_col * t
    decay = np.exp(-m_d_col * t)

    G_cos = np.cos(arg) * decay
    G_sin = np.sin(arg) * decay

    G = np.vstack((G_cos, G_sin))  # Shape: (m, N)

    # 2. Populate Gram matrix (Bretthorst Eq. 3.4).
    g = G @ G.T  # Shape: (m, m)

    # 3. Find eigenvalues and eigenvectors (Bretthorst 33).
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    # 4. Find orthonormal basis functions (Bretthorst Eq. 3.5).
    inv_sqrt_eig = 1.0 / np.sqrt(eigenvalues)
    H = (eigenvectors.T * inv_sqrt_eig[:, np.newaxis]) @ G

    # 5. Find projections of data onto orthonormal basis functions, orthonormal amplitudes (Bretthorst Eq. 3.13).
    h = H @ d

    # Pre-calculate dot products for performance
    d_sq_sum = np.dot(d, d)
    h_sq_sum = np.dot(h, h)

    # 6. Calculate hbar (Bretthorst Eq. 3.15).
    hbar = h_sq_sum / m

    # 7. Find probability (Bretthorst Eq. 3.17).
    ratio = (m * hbar) / d_sq_sum 
    log_probability = 0.5 * (m - N) * np.log10(1 - ratio)

    # 8. Find estimated variance (Bretthorst Eq. 4.7).
    variance = (d_sq_sum - h_sq_sum) / (N - m - 2)

    # 9. Find SNR (Bretthorst Eq. 4.8).
    SNR = np.sqrt((m / N) * (1 + hbar / variance))

    return log_probability, variance, SNR, h, H


def compute_uncertainties(t, d, model_frequencies, model_quality_factors):
    n_modes = len(model_frequencies)
    m = 2 * n_modes
    b = np.zeros((m, m))

    # 1. Populate the Hessian matrix (b) for all 2n parameters
    for ind_1 in range(m):
        for ind_2 in range(m):
            f_params = list(model_frequencies)
            q_params = list(model_quality_factors)

            # Baseline probability
            _, _, _, h_1, _ = compute_probability(t, d, f_params, q_params)
            hbar_1 = (1 / m) * np.sum(h_1 ** 2)

            # Perturb parameter 1
            if ind_1 < n_modes:
                delta_1 = f_params[ind_1] * 0.001 # 0.1% shift
                f_params[ind_1] += delta_1
            else:
                delta_1 = q_params[ind_1 - n_modes] * 0.001
                q_params[ind_1 - n_modes] += delta_1

            # Perturb parameter 2
            if ind_2 < n_modes:
                delta_2 = f_params[ind_2] * 0.001
                f_params[ind_2] += delta_2
            else:
                delta_2 = q_params[ind_2 - n_modes] * 0.001
                q_params[ind_2 - n_modes] += delta_2

            # Probability after perturbation
            _, _, _, h_2, _ = compute_probability(t, d, f_params, q_params)
            hbar_2 = (1 / m) * np.sum(h_2 ** 2)

            # Hessian calculation
            b[ind_1, ind_2] = -(m / 2) * (hbar_2 - hbar_1) / (delta_1 * delta_2)
            
    # 2. Get noise variance
    _, variance, _, _, _ = compute_probability(t, d, model_frequencies, model_quality_factors)

    # 3. Eigen-decomposition
    eigenvalues, eigenvectors = np.linalg.eig(b)

    # 4. Calculate uncertainties for ALL m parameters
    # total_uncertainties[0:n] = Freq uncertainties
    # total_uncertainties[n:2n] = Q factor uncertainties
    total_uncertainties = np.zeros(m)

    for ind in range(m):
        model_variance_sum = 0
        for j in range(m):
            # Variance propagation using the Moore-Penrose pseudo-inverse logic
            model_variance_sum += (eigenvectors[ind, j] ** 2) / np.abs(eigenvalues[j])

        total_uncertainties[ind] = np.sqrt(model_variance_sum * variance)

    # Split the results back into two arrays
    freq_uncertainties = total_uncertainties[:n_modes]
    q_uncertainties = total_uncertainties[n_modes:]

    return freq_uncertainties, q_uncertainties


if __name__ == "__main__":
    t = np.linspace(0, 360000 * 2, 10000)
    f1 = 0.000299
    f2 = 0.000301
    quality_factor_1 = 440
    quality_factor_2 = 600

    rng = np.random.default_rng(42)
    e = rng.uniform(low=-20, high=20, size=len(t))

    decay_rate_1 = 2 * np.pi * f1 / (2 * quality_factor_1)
    decay_rate_2 = 2 * np.pi * f2 / (2 * quality_factor_2)

    print(decay_rate_1, decay_rate_2)

    # Clean vs noisy data
    d_clean = (np.cos(2 * np.pi * f1 * t) + 2 * np.sin(2 * np.pi * f1 * t)) * np.e ** (-decay_rate_1 * t) + \
              (3 * np.cos(2 * np.pi * f2 * t) + np.sin(2 * np.pi * f2 * t)) * np.e ** (-decay_rate_2 * t)
    d = d_clean + e
    
    f1_grid = np.linspace(0.0002980, 0.0003020, 20)
    f2_grid = np.linspace(0.0002980, 0.0003020, 20)
    qf1_grid = np.linspace(200, 1000, 20)
    qf2_grid = np.linspace(200, 1000, 20)

    # 4D Array to track all probability scores across the parameter space
    log_prob_grid = np.zeros((len(f1_grid), len(qf1_grid), len(f2_grid), len(qf2_grid)))

    best_match = None
    max_log_prob = -np.inf

    pbar = tqdm.tqdm(total=len(f1_grid) * len(f2_grid) * len(qf1_grid) * len(qf2_grid))
    # Grid search (without storing large H matrices for every step)
    for i, freq1 in enumerate(f1_grid):
        for j, qf1 in enumerate(qf1_grid):
            for k, freq2 in enumerate(f2_grid):
                for l, qf2 in enumerate(qf2_grid):
                    log_prob, variance, SNR, h, H = compute_probability(t, d, [freq1, freq2], [qf1, qf2])
                    log_prob_grid[i, j, k, l] = log_prob
                    
                    if log_prob > max_log_prob:
                        max_log_prob = log_prob
                        best_match = (log_prob, variance, SNR, h, H, freq1, freq2, qf1, qf2)
                    pbar.update(1)

    # Convert log probabilities to normalized probabilities to prevent underflow
    prob_grid = np.exp(log_prob_grid - max_log_prob)
    prob_grid /= np.sum(prob_grid)

    # ==============================
    # PLOT 1: Corner Plot
    # ==============================
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    labels = ['f1', 'Q1', 'f2', 'Q2']
    grids = [f1_grid, qf1_grid, f2_grid, qf2_grid]

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                # 1D Marginal probability
                axes_to_sum = tuple(x for x in range(4) if x != i)
                marginal_1d = np.sum(prob_grid, axis=axes_to_sum)
                ax.plot(grids[i], marginal_1d, color='blue', drawstyle='steps-mid')
                ax.set_yticklabels([])
            elif i > j:
                # 2D Marginal probability
                axes_to_sum = tuple(x for x in range(4) if x != i and x != j)
                marginal_2d = np.sum(prob_grid, axis=axes_to_sum)
                X, Y = np.meshgrid(grids[j], grids[i])
                ax.contourf(X, Y, marginal_2d.T, cmap='Blues', levels=20)
            else:
                ax.axis('off')

            if i == 3 and j <= i:
                ax.set_xlabel(labels[j])
                ax.xaxis.set_major_locator(MaxNLocator(4))
            else:
                if i != j:
                    ax.set_xticklabels([])

            if j == 0 and i > 0:
                ax.set_ylabel(labels[i])
                ax.yaxis.set_major_locator(MaxNLocator(4))
            else:
                if i != j:
                    ax.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('corner_plot.png', dpi=150)
    plt.close()

    # ==============================
    # COMBINED ANALYSIS PLOT
    # ==============================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left Plot: FFT Amplitude Spectrum ---
    freqs, power = compute_fft(t, d, 0.000295, 0.000305)
    ax1.plot(freqs, np.abs(power), color='black', label='Data FFT', linewidth=1)
    ax1.axvline(best_match[5], color='red', linestyle='--', label=r'Estimated $f_1$')
    ax1.axvline(best_match[6], color='blue', linestyle='--', label=r'Estimated $f_2$')
    ax1.axvline(f1, color='orange', linestyle=':', alpha=0.8, label=r'True $f_1$')
    ax1.axvline(f2, color='cyan', linestyle=':', alpha=0.8, label=r'True $f_2$')
    
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Power')
    ax1.set_xlim(min(freqs), max(freqs))
    ax1.set_title('FFT of Data')
    ax1.legend(loc='upper right', fontsize='small')

    # --- Right Plot: Best Fit vs Original (Time Domain) ---
    h_best = best_match[3]
    H_best = best_match[4]
    reconstruction = H_best.T @ h_best

    # Cropped to 1000 points to show clear wave structure
    subset = slice(0, 1000) 
    ax2.plot(t[subset], d[subset], label='Data (with noise)', color='gray', alpha=0.5)
    ax2.plot(t[subset], d_clean[subset], label='True Signal', color='black', linewidth=1.5)
    ax2.plot(t[subset], reconstruction[subset], label='Model Reconstruction', color='red', linestyle='--', alpha=0.9)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Counts')
    ax2.set_title('Two Frequencies with Noise')
    ax2.set_xlim(t[subset].min(), t[subset].max())
    ax2.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig('combined_analysis.png', dpi=300)
    plt.show()

    # --- Output Stats ---
    print(f"Optimal Parameters: f1={best_match[5]:.9f}, f2={best_match[6]:.9f}, Q1={best_match[7]:.1f}, Q2={best_match[8]:.1f}")
    
    f_uncertainties, q_uncertainties = compute_uncertainties(t, d, [best_match[5], best_match[6]], [best_match[7], best_match[8]])
    print(f"Freq Uncertainties: {f_uncertainties}")
    print(f"Q Uncertainties: {q_uncertainties}")

    alpha_uncertainties = [np.pi * best_match[5] / best_match[7], np.pi * best_match[6] / best_match[8]]
    print(f"Alpha Uncertainties: {alpha_uncertainties}")
    print(f"Noise Variance: {best_match[1]:.4f} | SNR: {best_match[2]:.4f}")