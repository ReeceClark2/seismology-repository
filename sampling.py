import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal, stats
import math
import random


def get_synthetic_data(filename, minimum_frequency=None, maximum_frequency=None):
    data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')

    # Column 0 is time in seconds, Column 1 is intensity
    t = data.iloc[:, 0].values
    d = data.iloc[:, 1].values
    
    if minimum_frequency and maximum_frequency:
        # Calculate actual delta from the file data
        delta = np.mean(np.diff(t)) 
        fs = 1.0 / delta

        nyquist = 0.5 * fs
        low = minimum_frequency / nyquist
        high = maximum_frequency / nyquist

        order = 4
        b, a = signal.butter(order, [low, high], btype='band')

        # Detrending prevents edge offsets from blowing up the filter
        detrended = signal.detrend(d)

        d = signal.filtfilt(b, a, detrended)

    return t, d


def compute_fft(t, d, minimum_frequency, maximum_frequency):
    delta = np.mean(np.diff(t)) 

    taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
    d = d * taper
    
    nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 2)
    
    frequencies = np.fft.fftfreq(n=nfft, d=delta)
    power = np.fft.fft(d, n=nfft) * delta

    mask = (frequencies >= minimum_frequency) & (frequencies <= maximum_frequency)
    return frequencies[mask], power[mask]


def find_peaks(x, y, threshold=None):
    if not threshold:
        threshold = 100 * np.median(y)

    deltas = np.diff(y)

    peak_indices = []

    N = len(deltas) - 1
    for ind in range(N):
        if deltas[ind] > 0 and deltas[ind + 1] < 0 and y[ind + 1] > threshold:
            peak_indices.append(ind + 1)

    peak_x = []
    peak_y = []
    for ind in peak_indices:
        peak_x.append(x[ind])
        peak_y.append(y[ind])

    return peak_x, peak_y


def compute_probability(t, d, model_frequencies, model_quality_factors):
    model_frequencies = 2 * np.pi * np.array(model_frequencies)
    model_decay_rates = model_frequencies / model_quality_factors

    r = len(model_frequencies)
    m = 2 * r
    N = len(d)

    # Initialize functions and Gram matrix (Bretthorst Eq. 3.1 & Eq. 3.4).
    G = np.zeros(m, dtype=object)
    g = np.zeros((m, m))

    # Populate model function arrays (Bretthorst page 32).
    for j in range(r):
            G[j] = np.cos(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)
            G[j + r] = np.sin(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)

    # Populate Gram matrix (Bretthorst Eq. 3.4).
    for j in range(r):
        for k in range(r):
            g[j, k] = np.sum(G[j] * G[k])
            g[j + r, k] = np.sum(G[j + r] * G[k])
            g[j, k + r] = np.sum(G[j] * G[k + r])
            g[j + r, k + r] = np.sum(G[j + r] * G[k + r])

    # Find eigenvalues and eigenvectors (Bretthorst 33).
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    eigenvalues = np.maximum(eigenvalues, 1e-8)

    # Find orthonormal basis functions (Bretthorst Eq. 3.5).
    H = np.zeros(m, dtype=object)
    for j, _ in enumerate(H):
        for k in range(m):
            H[j] += (1 / np.sqrt(eigenvalues[j])) * eigenvectors[k][j] * G[k]

    # Find projections of data onto orthonormal basis functions, orthonormal amplitudes (Bretthorst Eq. 3.13).
    h = np.zeros(m)
    for j, _ in enumerate(h):
        h[j] = np.sum(d * H[j])

    # Calculate dbar (Bretthorst page 17).
    dbar = (1 / N) * np.sum(d ** 2)

    # Calculate hbar (Bretthorst Eq. 3.15).
    hbar = (1 / m) * np.sum(h ** 2)

    # Find probability (Bretthorst Eq. 3.17).
    ratio = (m * hbar) / (N * dbar)
    log_probability = 0.5 * (m - N) * np.log10(1 - ratio)

    # Find estimated variance (Bretthorst Eq. 4.7).
    variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))

    # Find SNR (Bretthorst Eq. 4.8).
    SNR = ((m / N) * (1 + hbar / variance)) ** (0.5)

    return log_probability, variance, SNR, h, H


def metropolis_hastings(t, d, current_frequencies, current_quality_factors, 
                        iterations=10000, 
                        maximum_frequency_step=0.00001, 
                        maximum_quality_factor_step=10):
    
    current_probability, _, _, _, _ = compute_probability(t, d, current_frequencies, current_quality_factors)

    progress = tqdm.tqdm(total=iterations)
    for _ in range(iterations):
        proposal_frequencies = [current_frequency + random.uniform(-maximum_frequency_step, maximum_frequency_step) for current_frequency in current_frequencies]
        proposal_quality_factors = [current_quality_factor + random.uniform(-maximum_quality_factor_step, maximum_quality_factor_step) for current_quality_factor in current_quality_factors]

        proposal_probability, _, _, _, _ = compute_probability(t, d, proposal_frequencies, proposal_quality_factors)

        ratio = proposal_probability - current_probability

        c = random.random()
        log_c = np.log10(c)

        if log_c < ratio:
            current_frequencies = proposal_frequencies
            current_quality_factors = proposal_quality_factors

            current_probability = proposal_probability

        progress.update(1)

    return current_frequencies, current_quality_factors


# TODO: Find initial decay rates!
# def find_decay_rate(t, d):
#     peak_t, peak_d = find_peaks(t, d, threshold=0)

#     plt.scatter(peak_t, np.log(peak_d))

#     res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))
#     x = np.linspace(min(t), max(t), 100)
#     y = res.slope * x + res.intercept

#     print((2 * np.pi * 0.000305) / res.slope)


if __name__ == "__main__":
    plotting = input("Save plots? (Y/N) ")
    number_of_trials = input("Number of trials: ")

    for trial in range(number_of_trials):
        minimum_frequency = 0.0002
        maximum_frequency = 0.0005

        file_path = "timeseries_Russia/IU_HRV_TS.ascii" 

        t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
        f, p = compute_fft(t, d, minimum_frequency, maximum_frequency)

        peak_f, _ = find_peaks(f, abs(p))
        
        guess_frequencies = peak_f
        guess_quality_factors = [random.randint(200, 1000) for _ in range(len(peak_f))]

        new_f, new_q = metropolis_hastings(t, d, guess_frequencies, guess_quality_factors, iterations=10000)

        _, old_variance, old_SNR, old_h, old_H = compute_probability(t, d, guess_frequencies, guess_quality_factors)
        _, new_variance, new_SNR, new_h, new_H = compute_probability(t, d, new_f, new_q)

        if plotting == "Y":
            old_model = np.zeros(len(t))
            new_model = np.zeros(len(t))
            for ind, _ in enumerate(old_h):
                old_model += old_h[ind] * old_H[ind]
                new_model += new_h[ind] * new_H[ind]

            fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

            axes[0].plot(t, d, color="seagreen", alpha=0.6)
            axes[0].set_xlim(min(t), max(t))
            axes[0].set_title("Original Data")

            axes[1].plot(t, old_model, color="cornflowerblue", alpha=0.8)
            axes[1].plot(t, old_model - d, color="red", alpha=0.2)
            axes[1].set_xlim(min(t), max(t))
            axes[1].set_title("Old Model")

            axes[2].plot(t, new_model, color="sandybrown", alpha=0.8)
            axes[2].plot(t, new_model - d, color="red", alpha=0.2)
            axes[2].set_xlim(min(t), max(t))
            axes[2].set_title("New Model")

            for ax in axes:
                ax.set_xlabel("Time")

            plt.tight_layout()
            plt.savefig("model_comparison.png", dpi=300)


        with open(f"modeling_run_{trial}.txt", "w") as f_out:
            f_out.write("================================================\n")
            f_out.write("          METROPOLIS-HASTINGS RUN SUMMARY       \n")
            f_out.write("================================================\n\n")

            f_out.write("--- INITIAL PARAMETERS ---\n")
            f_out.write(f"Source File:      {file_path}\n")
            f_out.write(f"Frequency Range:  {minimum_frequency} to {maximum_frequency} Hz\n")
            f_out.write(f"Number of Peaks:  {len(guess_frequencies)}\n\n")

            f_out.write("--- MODEL COMPARISON ---\n")
            f_out.write(f"{'Metric':<20} | {'Initial Guess':<15} | {'Optimized'}\n")
            f_out.write("-" * 55 + "\n")
            f_out.write(f"{'Variance':<20} | {old_variance:<15.4e} | {new_variance:.4e}\n")
            f_out.write(f"{'SNR':<20} | {old_SNR:<15.4f} | {new_SNR:.4f}\n\n")

            f_out.write("--- DETECTED MODES ---\n")
            f_out.write(f"{'Mode #':<8} | {'Initial Freq':<15} | {'Final Freq':<15} | {'Final Q'}\n")
            f_out.write("-" * 60 + "\n")
            
            for i in range(len(new_f)):
                f_out.write(f"{i+1:<8} | {guess_frequencies[i]:<15.6e} | {new_f[i]:<15.6e} | {new_q[i]:.2f}\n")

            f_out.write("\n--- END OF LOG ---\n")


