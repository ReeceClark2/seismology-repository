import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import tqdm
import pandas as pd
from scipy import signal
import math
from BATS4 import BATS


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

    t = t[144:]
    d = d[144:]

    return t, d

def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')

    inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
    stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

    trace = stream[stream_index]

    trace.detrend('constant')

    trace.remove_response(inventory=inventory, output="ACC")

    if minimum_frequency and maximum_frequency:
        trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
        
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    delta = trace.stats.delta 
    N = len(trace)
    t = np.arange(N) * delta
    
    d = np.array(trace.data)

    t = t[144:]
    d = d[144:]

    return t, d

minimum_frequency = 0.0005     # Minimum frequency (Hz)
maximum_frequency = 0.0012     # Maximum frequency (Hz)

network = "IU"                  # Network
station = "KIP"                 # Station
channel = "LHZ"                 # Channel
location = "00"                 # Location

stream_index = 0                                    # Stream index
start_time = UTCDateTime('2025-07-31T06:24:50')     # Start time
end_time = UTCDateTime('2025-08-11T05:24:50')       # End time

file_path = "timeseries_Russia/G_FDF_TS.ascii" 

# t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)
# d = np.sin(2 * np.pi * 0.0011 * t) + np.sin(2 * np.pi * 0.0007 * t) + np.sin(2 * np.pi * 0.0009 * t)


model = BATS(t, d, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
model.run(lower_bound_model_functions=13, upper_bound_model_functions=25)
