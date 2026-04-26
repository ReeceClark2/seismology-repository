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


def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')

    # Use obspy to retrieve data
    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

    # Run bandpass on retrieved data
    trace = stream[stream_index]
    trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
    trace.decimate(5, no_filter=False)
    trace.decimate(4, no_filter=False)
    trace.detrend('constant')

    # Create accurate time array assuming evenly sampled data
    delta = trace.stats.delta 
    N = len(trace)
    t = np.arange(N) * delta

    return t, np.array(trace)


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
        threshold = 10 * np.median(y)

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

    # plt.plot(f, abs(p), color="cornflowerblue", alpha=0.8)
    # plt.title("HRV, ANMO, KONO")
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Power")

    # plt.xlim(min(f), max(f))
    # plt.plot([min(f), max(f)], np.ones(2) * threshold)
    # plt.show()

    return peak_x, peak_y


def find_quality_factor(t, d, frequency):
    peak_t, peak_d = find_peaks(t, d, threshold=0)

    res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

    quality_factor = np.pi * frequency / (-1 * res.slope)

    return quality_factor


def find_quality_factors(t, d, guess_frequencies):
    quality_factors = []

    for guess_frequency in guess_frequencies:
        minimum_frequency = guess_frequency - 0.00003
        maximum_frequency = guess_frequency + 0.00003

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

        frequency_d = signal.filtfilt(b, a, detrended)

        peak_t, peak_d = find_peaks(t, frequency_d, threshold=0)

        res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

        plt.figure(figsize=(10, 6))
        plt.plot(t, frequency_d, color='gray', label='Data', linewidth=0.1)
        plt.scatter(peak_t, peak_d, color='red', s=0.4)
        plt.plot(t, max(frequency_d) * np.e ** (res.slope * t), color='red', linewidth=0.3)
        
        plt.xlim(min(t), max(t))

        plt.xlabel("Time (s)")
        plt.ylabel("Displacement")
        plt.title("Quality Factor Initial Measurement")
        plt.legend()

        plt.savefig(f"sampling_results/decay_envelope_{guess_frequency}.png", dpi=300)
        plt.close("all")

        quality_factor = np.pi * guess_frequency / (-1 * res.slope)
        
        quality_factors.append(quality_factor)


    return quality_factors


def compute_probability(t, d, model_frequencies, model_quality_factors):
    model_frequencies = 2 * np.pi * np.array(model_frequencies)
    model_decay_rates = model_frequencies / (2 * np.array(model_quality_factors))

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


def metropolis_hastings(t, d, current_frequencies, current_quality_factors, trial, 
                        iterations=10000, 
                        maximum_frequency_step=0.0000001, 
                        maximum_quality_factor_step=10):
    
    current_probability, _, _, _, _ = compute_probability(t, d, current_frequencies, current_quality_factors)
    accepted = 0

    probabilities = []

    progress = tqdm.tqdm(total=iterations)
    for _ in range(iterations):
        proposal_frequencies = [current_frequency + random.uniform(-maximum_frequency_step, maximum_frequency_step) for current_frequency in current_frequencies]
        proposal_quality_factors = [current_quality_factor + random.uniform(-maximum_quality_factor_step, maximum_quality_factor_step) for current_quality_factor in current_quality_factors]

        proposal_probability, _, _, _, _ = compute_probability(t, d, proposal_frequencies, proposal_quality_factors)
        if proposal_probability is None:
            continue

        ratio = proposal_probability - current_probability

        c = random.random()
        log_c = np.log10(c)

        if log_c < ratio:
            current_frequencies = proposal_frequencies
            current_quality_factors = proposal_quality_factors

            current_probability = proposal_probability

            accepted += 1

        probabilities.append(current_probability)
        progress.update(1)

    plt.plot(probabilities, color="gray")

    plt.xlim(0, iterations)
    plt.xlabel("Iteration")
    plt.ylabel(r"Log$_{10}$(Probability)")
    plt.title("Metropolis Hastings Sampling")

    plt.savefig(f"sampling_results/probabilities_{trial}.png", dpi=300)

    print("Acceptance rate: ", 100 * accepted / iterations)

    return current_frequencies, current_quality_factors


if __name__ == "__main__":
    number_of_trials = int(input("Number of trials: "))

    for trial in range(number_of_trials):
        minimum_frequency = 0.00020     # Minimum frequency (Hz)
        maximum_frequency = 0.00060     # Maximum frequency (Hz)

        network = "IU"                  # Network
        station = "HRV"                # Station
        channel = "LHZ"                 # Channel
        location = "00"                 # Location

        stream_index = 0                                    # Stream index
        start_time = UTCDateTime('2025-07-31T06:24:50')     # Start time
        end_time = UTCDateTime('2025-08-11T05:24:50')       # End time

        file_path = f"timeseries_Russia/{network}_{station}_TS.ascii" 

        t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
        # t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
        f, p = compute_fft(t, d, minimum_frequency, maximum_frequency)

        Path("sampling_results").mkdir(exist_ok=True)

        peak_f, _ = find_peaks(f, abs(p))
        
        guess_frequencies = peak_f
        guess_quality_factors = find_quality_factors(t, d, guess_frequencies)

        new_f, new_q = metropolis_hastings(t, d, guess_frequencies, guess_quality_factors, trial, iterations=10000)

        _, old_variance, old_SNR, old_h, old_H = compute_probability(t, d, guess_frequencies, guess_quality_factors)
        _, new_variance, new_SNR, new_h, new_H = compute_probability(t, d, new_f, new_q)

        old_model = np.zeros(len(t))
        new_model = np.zeros(len(t))
        for ind, _ in enumerate(old_h):
            old_model += old_h[ind] * old_H[ind]
            new_model += new_h[ind] * new_H[ind]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

        axes[0].plot(t, d, color="seagreen", alpha=0.6)
        axes[0].set_xlim(min(t), max(t))
        axes[0].set_ylim(min(d) * 1.1, -1 * min(d) * 1.1)
        axes[0].set_title("Original Data")

        axes[1].plot(t, old_model, color="cornflowerblue", alpha=0.8)
        axes[1].plot(t, old_model - d, color="red", alpha=0.2)
        axes[1].set_xlim(min(t), max(t))
        axes[1].set_ylim(min(d) * 1.1, -1 * min(d) * 1.1)
        axes[1].set_title("Old Model")

        axes[2].plot(t, new_model, color="sandybrown", alpha=0.8)
        axes[2].plot(t, new_model - d, color="red", alpha=0.2)
        axes[2].set_xlim(min(t), max(t))
        axes[2].set_ylim(min(d) * 1.1, -1 * min(d) * 1.1)
        axes[2].set_title("New Model")

        for ax in axes:
            ax.set_xlabel("Time")

        plt.tight_layout()
        plt.savefig(f"sampling_results/model_comparison_time_series_{trial}.png", dpi=300)
        plt.close("all")

        plt.figure(figsize=(10, 6))
        plt.plot(f, abs(p), label="FFT of Data", color="gray", linewidth=0.8)
        plt.vlines(guess_frequencies, -100, 1000, label="Guess Frequencies", color="cornflowerblue", alpha=0.8, linewidth=0.45)
        plt.vlines(new_f, -100, 1000, label="New Frequencies", color="sandybrown", alpha=0.8, linewidth=0.45)

        plt.ylim(0, max(abs(p)) * 1.1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("Frequency Comparison")
        plt.legend()

        plt.savefig(f"sampling_results/model_comparison_fourier_space_{trial}.png", dpi=500)


        with open(f"sampling_results/modeling_run_{trial}.txt", "w") as f_out:
            # Summary Metrics
            f_out.write(f"Source: {file_path}\n")
            f_out.write(f"Variance (Old -> New): {old_variance:.4e} -> {new_variance:.4e}\n")
            f_out.write(f"SNR (Old -> New): {old_SNR:.2f} -> {new_SNR:.2f}\n\n")

            # Column Headers
            f_out.write("Mode | Init_Freq | Final_Freq | Init_Q | Final_Q\n")
            f_out.write("-" * 50 + "\n")
            
            # Parameters for each mode
            for i in range(len(new_f)):
                f_out.write(f"{i+1} | {guess_frequencies[i]:.6e} | {new_f[i]:.6e} | {guess_quality_factors[i]:.2f} | {new_q[i]:.2f}\n")