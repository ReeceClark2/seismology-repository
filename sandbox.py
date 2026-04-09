import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal
import math


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


def find_peaks(frequencies, power, threshold=None):
    if not threshold:
        threshold = 100 * np.median(power)

    deltas = np.diff(power)

    peak_indices = []

    N = len(deltas) - 1
    for ind in range(N):
        if deltas[ind] > 0 and deltas[ind + 1] < 0 and power[ind + 1] > threshold:
            peak_indices.append(ind + 1)

    peak_frequencies = []
    for ind in peak_indices:
        peak_frequencies.append(frequencies[ind])

    return peak_frequencies


def metropolis



if __name__ == "__main__":
    minimum_frequency = 0.0002
    maximum_frequency = 0.0004

    file_path = "timeseries_Russia/IU_HRV_TS.ascii" 

    t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
    f, p = compute_fft(t, d, minimum_frequency, maximum_frequency)

    peak_frequencies = find_peaks(f, abs(p))

    print(peak_frequencies)
