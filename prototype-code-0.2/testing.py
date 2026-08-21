import csv
import math
import time

import numpy as np
import pandas as pd
from scipy import signal as sp_signal

from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

from BATS import BATS

import matplotlib.pyplot as plt


def get_synthetic_data(filename, min_f=None, max_f=None):
    data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')
    t = data.iloc[:, 0].values
    d = data.iloc[:, 1].values

    delta = np.mean(np.diff(t))
    fs = 1.0 / delta
    nyquist = 0.5 * fs
    low = min_f / nyquist
    high = max_f / nyquist

    b, a = sp_signal.butter(4, [low, high], btype='band')
    detrended = sp_signal.detrend(d)
    d = sp_signal.filtfilt(b, a, detrended)

    t = t[288:]
    d = d[1440:]
    return t, d


def get_observed_data(network, station, channel, location, stream_index,
                      start_time, end_time, min_f, max_f):
    client = Client('IRIS')
    inventory = client.get_stations(network=network, station=station,
                                    location=location, channel=channel,
                                    starttime=start_time, endtime=end_time,
                                    level='response')
    stream = client.get_waveforms(network=network, station=station,
                                  location=location, channel=channel,
                                  starttime=start_time, endtime=end_time)
    trace = stream[stream_index]
    trace.detrend('constant')
    trace.remove_response(inventory=inventory, output="ACC")
    if min_f and max_f:
        trace.filter('bandpass', freqmin=min_f, freqmax=max_f)
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



min_f = 0.00025
max_f = 0.00160

network = "IU"
station = "KIP"
channel = "LHZ"
location = "00"

stream_index = 0
start_time = UTCDateTime('2025-07-31T06:24:50')
end_time = UTCDateTime('2025-08-11T05:24:50')

file_path = f"../timeseries-kamchatka/{network}_{station}_TS.ascii"

data = "observed"

if data == "observed":
    t, d = get_observed_data(network, station, channel, location,
                                stream_index, start_time, end_time,
                                min_f, max_f)
else:
    t, d = get_synthetic_data(file_path, min_f, max_f)

f_bw = 1e-4 * np.ones(50)
k_bw = 1e-5 * np.ones(50)

model = BATS(    t, 
                 d, 
                 min_f=min_f, 
                 max_f=max_f,
                 k_type="linear",

                 mode="global",
                 min_signals=35,
                 max_signals=36,

                 sampler="NUTS",
                 burn_in=1,
                 typical_set=1,
                 acceptance=0.8,
                 dense_mass=True,
                 boundaries=True,
                 f_bw=f_bw,
                 k_bw=k_bw,
                 std=5)

model.launch()
