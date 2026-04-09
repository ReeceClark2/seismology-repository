from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import sys
import tqdm
import pandas as pd
import seaborn as sns

from observation_process import Observation_Process
from synthetic_process import Synthetic_Process


min_frequency = 0.00027 # Minimum frequency for FFT
max_frequency = 0.00033 # Maximum frequency for FFT
frequency_space = np.linspace(0.00028, 0.00034, 200)
quality_factor_space = np.linspace(510, 510, 1)

net = "IU"          # Network
sta = "ANMO"        # Station
chan = "LHZ"        # Channel
loc = "00"          # Location

stream_index = 0                                    # Stream index
start_time = UTCDateTime('2025-07-29T00:24:50')     # Start time
end_time = UTCDateTime('2025-08-09T05:24:50')       # End time

data = Observation_Process(net, sta, chan, loc, start_time, end_time)
d = data.create_time_series(min_frequency, max_frequency, 0)
d = (d - np.mean(d)) / np.max(np.abs(d))
delta = data.stream[0].stats.delta 
N = len(d)
t = np.arange(N) * delta
d = d[35000:]
t = t[35000:]
N -= 35000


# file = "timeseries_Russia/IU_HRV_TS.ascii"
# min_frequency = 0.1 # Minimum frequency for FFT
# max_frequency = 0.4 # Maximum frequency for FFT
# window = 400        # Window size in hours
# data = Synthetic_Process(file, 0, 0.0002, 0.0004)
# d = data.intensities / max(data.intensities)
# N = len(d)
# t = data.times
probabilities10 = []
frequencies = []
quality_factors = []
pbar = tqdm.tqdm(total=200)
for quality_factor in quality_factor_space:
    log10_probability_list = []
    for frequency in frequency_space:
        # Guess model frequencies
        model_frequencies = 2 * np.pi * np.array([frequency]) # Synthetic
        model_quality_factors = np.array([quality_factor])
        model_decay_rates = np.array(model_frequencies) / (2 * model_quality_factors)


        # Initalize model functions and Gram matrix.
        r = len(model_frequencies)
        m = 2 * r
        G = np.zeros(m, dtype=object)
        g = np.zeros((2 * r, 2 * r))

        # Populate model function arrays (Bretthorst page 32).
        for j in range(r):
            G[j] = np.cos(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)
            G[j + r] = np.sin(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)

        # Populate Gram matrix (Bretthorst page 32).
        for j in range(r):
            for k in range(r):
                g[j, k] = np.sum(G[j] * G[k])
                g[j + r, k] = np.sum(G[j + r] * G[k])
                g[j, k + r] = np.sum(G[j] * G[k + r])
                g[j + r, k + r] = np.sum(G[j + r] * G[k + r])

        # Find eigenvalues and eigenvectors (Bretthorst 33).
        eigenvalues, eigenvectors = np.linalg.eigh(g)

        # print("Eigenvalues:\n", eigenvalues)
        # print("\nEigenvectors:\n", eigenvectors)
        # print("\ng matrix:\n", g)

        # Find orthonormal basis functions (Bretthorst Eq. 3.5).
        H = np.zeros(m, dtype=object)
        for j, _ in enumerate(H):
            for k in range(m):
                H[j] += (1 / np.sqrt(eigenvalues[j])) * eigenvectors[k][j] * G[k]

        # Find projections of data onto orthonormal basis functions, orthonormal amplitudes (Bretthorst Eq. 3.13).
        h = np.zeros(m)
        for j, _ in enumerate(h):
            h[j] = np.sum(d * H[j])

        # Create model function from guess parameters.
        model = np.zeros(N)
        for j, _ in enumerate(h):
            model += h[j] * H[j]

        # Calculate dbar (Bretthorst page 17).
        dbar = (1 / N) * np.sum(d ** 2)

        # Calculate hbar (Bretthorst Eq. 3.15).
        hbar = (1 / m) * np.sum(h ** 2)

        # Find probability (Bretthorst Eq. 3.17).
        ratio = (m * hbar) / (N * dbar)
        log10_probability = 0.5 * (m - N) * np.log10(1 - ratio)

        # print("\nLog10 probability: ", round(log10_probability, 4))

        # # Find estimated variance (Bretthorst Eq. 4.7).
        # variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))
        # print("Estimated variance: ", round(variance, 4))

        # # Find SNR (Bretthorst Eq. 4.8).
        # SNR = ((m / N) * (1 + hbar / variance)) ** (0.5)
        # print("SNR: ", round(SNR, 4))
        
        log10_probability_list.append(log10_probability)
        pbar.update(1)
        frequencies.append(frequency)
        quality_factors.append(quality_factor)
        probabilities10.append(log10_probability)


power, frequency = data.create_spectrum(min_frequency * 1000, max_frequency * 1000, start_time, end_time, 0)

plt.plot(frequency_space, probabilities10)

# frame = pd.DataFrame({
#     'Frequency': frequencies,
#     'Quality': quality_factors,
#     'Probability': probabilities10
# })

# heatmap_data = frame.pivot(index='Quality', columns='Frequency', values='Probability')
# sns.heatmap(heatmap_data, cmap='viridis')
plt.show()

# plt.plot(t, d, color="sandybrown", alpha=0.6, label="Synthetic Data")
# plt.plot(t, model, color="cornflowerblue", alpha=0.6, label="Model")
# # plt.plot(t, d - model, color="seagreen", alpha=0.75, label="Residuals")

# plt.xlim(min(t), max(t))
# plt.xlabel("Time (s)")
# plt.ylabel("Intensity")
# plt.title("Bayesian Approach to 0S2 Normal Mode")
# plt.legend()
# plt.show()
