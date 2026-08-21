import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal
import math


def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')

    # Use obspy to retrieve data
    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

    # Run bandpass on retrieved data
    trace = stream[stream_index]
    trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
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

    return t, d


def run_student_t_distribution(t, d, model_frequencies, model_decay_rates=None):
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
    log10_probability = 0.5 * (m - N) * np.log10(1 - ratio)

    # Find estimated variance (Bretthorst Eq. 4.7).
    variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))

    # Find SNR (Bretthorst Eq. 4.8).
    SNR = ((m / N) * (1 + hbar / variance)) ** (0.5)

    return log10_probability, h, H, variance, SNR


def locate_parameters(t, d, total_functions, minimum_frequency, maximum_frequency, frequency_sampling, minimum_quality_factor, maximum_quality_factor, quality_factor_sampling):
    # Define frequency and quality factor sampling space. Convert frequency to angular frequency (Bretthorst page 22).
    frequency_sampling_space = 2 * np.pi * np.linspace(minimum_frequency, maximum_frequency, frequency_sampling)
    quality_factor_space = np.linspace(minimum_quality_factor, maximum_quality_factor, quality_factor_sampling)

    # Initialize function parameters to be found.
    located_frequencies = []
    located_decay_rates = []


    # Iterate over number of functions to be found.
    for i in range(total_functions):
        outcomes = []

        # Iterate over parameter space to locate most likely parameters.
        pbar = tqdm.tqdm(total=frequency_sampling * quality_factor_sampling, desc=f"Sampling Function {i + 1}")
        for frequency in frequency_sampling_space:
            for quality_factor in quality_factor_space:
                decay_rate = frequency / (2 * quality_factor)

                log10_probability, h, H, _, _ = run_student_t_distribution(t, d, np.array([frequency]), np.array([decay_rate]))

                outcomes.append([frequency / (2 * np.pi), decay_rate, log10_probability, h, H])

                pbar.update(1)

        # Save parameters for maximum probability.
        max_index = max(range(len(outcomes)), key=lambda i: outcomes[i][2])

        located_frequencies.append(outcomes[max_index][0])
        located_decay_rates.append(outcomes[max_index][1])

        # Subtract best fit model from data to retrieve residuals.
        h = outcomes[max_index][3]
        H = outcomes[max_index][4]
        for j, _ in enumerate(h):
            d -= h[j] * H[j]

        frequencies = [row[0] for row in outcomes]
        log_probs = [row[2] for row in outcomes]

        # Set max probability on first iteration to normalize plot.
        if i == 0:
            max_probability = max(log_probs)

        # Plotting for each run.
        plt.figure(figsize=(12, 7))
        plt.plot(frequencies, log_probs / max_probability, color="cornflowerblue", alpha=0.8, label=f"{i} Order Residuals")
        plt.legend()
        
        plt.xlim(minimum_frequency, maximum_frequency)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Log10 Probability")
        plt.title(r"$_0S_2$ Modeling")
        
        plt.savefig(f"Function{i}.png", dpi=300)

        # TODO: Before next iteration check whether n function model is more likely than n-1 function model. If not, then break loop.

    return located_frequencies, located_decay_rates



if __name__ == "__main__":
    minimum_frequency = 0.00028     # Minimum frequency (Hz)
    maximum_frequency = 0.00034     # Maximum frequency (Hz)

    network = "IU"                  # Network
    station = "ANMO"                # Station
    channel = "LHZ"                 # Channel
    location = "00"                 # Location

    stream_index = 0                                    # Stream index
    start_time = UTCDateTime('2025-07-31T06:24:50')     # Start time
    end_time = UTCDateTime('2025-08-11T05:24:50')       # End time

    file_path = "timeseries_Russia/G_FDF_TS.ascii" 

    # t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
    t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)

    total_functions = 5             # Total functions attempted to define
    frequency_sampling = 1000       # Total frequencies to sample between the minimum and maximum defined frequencies

    minimum_quality_factor = 510    # Minimum quality factor to sample
    maximum_quality_factor = 510    # Maximum quality factor to sample
    quality_factor_sampling = 1     # Total quality factors to sample between the minimum and maximum defined quality factors

    located_frequencies, located_decay_rates = locate_parameters(t, d.copy(), total_functions, minimum_frequency, maximum_frequency, frequency_sampling, minimum_quality_factor, maximum_quality_factor, quality_factor_sampling)
    prob, h, H, variance, SNR = run_student_t_distribution(t, d, 2 * np.pi * np.array(located_frequencies), model_decay_rates=np.array(located_decay_rates))

    plt.figure(figsize=(12, 7))
    plt.plot(t, d, color="sandybrown", alpha=0.8, label="Observed")
    
    # Create model function from guess parameters.
    model = np.zeros(len(t))
    for j, _ in enumerate(h):
        model += h[j] * H[j]
    
    # Plotting for data, model, and residuals.
    plt.plot(t, model, color="cornflowerblue", alpha=0.8, label="Model")
    plt.plot(t, d - model, color="seagreen", alpha=0.4, label="Residuals")
    plt.legend()

    plt.xlim(min(t), max(t))
    plt.xlabel("Time (s)")
    plt.ylabel("Intensity")
    plt.title(r"$_0S_2$ Modeling Time Series")

    plt.show()
    