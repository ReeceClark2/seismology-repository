#!/usr/bin/env python
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from synthetic_process import Synthetic_Process
from observation_process import Observation_Process
import utils


file = "timeseries_Russia/IU_HRV_TS.ascii"  # Synthetic time series to use
synthetic = Synthetic_Process(file, 0)

min_frequency = 0.2 # Minimum frequency for FFT
max_frequency = 1.2 # Maximum frequency for FFT

window = 60

synthetic_start_time = 0
data_start_time = UTCDateTime('2025-07-29T23:24:50') # Start time
data_end_time = UTCDateTime('2025-08-6T23:24:50')    # End time

length = 360        # TODO: What is length?
net = "IU"          # Network
sta = "HRV"         # Station
chan = "LHZ"        # Channel
loc = "00"          # Location

stream_index = 0

data = Observation_Process(length, net, sta, chan, loc, data_start_time, data_end_time)

xs = []
ys = []

pbar = tqdm.tqdm(100)
for i in range(100):
    data_start_time += 3600
    data_end_time = data_start_time + (window * 3600)

    synthetic_start_time += 1

    power1, frequency1 = synthetic.create_spectrum(min_frequency / 1000, max_frequency / 1000, window, synthetic_start_time)

    # power2, frequency2 = data.create_spectrum(min_frequency, max_frequency, data_start_time, data_end_time, stream_index)

    xs.append([frequency1 * 1000])
    ys.append([abs(power1) / max(abs(power1))])
    # xs.append([frequency1 * 1000, frequency2])
    # ys.append([abs(power1) / max(abs(power1)), abs(power2) / max(abs(power2))])
    pbar.update(1)

utils.animate(xs, ys, labels=["Synthetic", "Data"], colors=["navajowhite", "lightsteelblue"], xlabel="Frequency (mHz)", ylabel="Power", title="Normal Mode Spectra")
    