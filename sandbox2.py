from BATS import BATS
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

    # Extract the specific trace
    trace = stream[stream_index]

    # 1. Detrend FIRST. Removing response from data with a DC offset causes artifacts.
    trace.detrend('constant')

    # 2. Remove instrument response and output as Acceleration (m/s^2)
    trace.remove_response(inventory=inventory, output="ACC")

    # 3. Filter and decimate
    if minimum_frequency and maximum_frequency:
        trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
        
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    # Create accurate time array assuming evenly sampled data
    delta = trace.stats.delta 
    N = len(trace)
    t = np.arange(N) * delta
    
    # Extract the data array
    d = np.array(trace.data) 

    # Trim the first 500 samples
    t = t[144:]
    d = d[144:]

    return t, d


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

    d = d[144:]
    t = t[144:]
    
    return t, d


minimum_frequency = 0.00025     # Minimum frequency (Hz)
maximum_frequency = 0.00055    # Maximum frequency (Hz)

network = "IU"                  # Network
station = "KIP"                 # Station
channel = "LHZ"                 # Channel
location = "00"                 # Location

stream_index = 0                                    # Stream index
start_time = UTCDateTime('2025-07-29T23:24:52')     # Origin time (UTC) of M8.8 event
end_time = start_time + (15 * 24 * 60 * 60)         # Exactly 15 days after (2025-08-13T23:24:52)

file_path = f"timeseries_Russia/{network}_{station}_TS.ascii" 

t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
# t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)

model = BATS(t, d, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency, threshold=8)
model.run(iterations=10_000)