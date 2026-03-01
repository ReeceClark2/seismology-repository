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
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from synthetic_process import Synthetic_Process
from observation_process import Observation_Process
import utils


folder = "timeseries_Russia/"
pbar = tqdm.tqdm(total=len(list(Path(folder).iterdir())))

frequencies = []
data_power = []
synthetic_power = []
networks = []
stations = []

for file_path in Path(folder).iterdir():
    try:
        match = re.search(r"\\([^_]+)_([^_.]+)", str(file_path))
        if match:
            net = match.group(1)
            sta = match.group(2)

            synthetic = Synthetic_Process(str(file_path), 0)

            min_frequency = 0.2 # Minimum frequency for FFT
            max_frequency = 1.2 # Maximum frequency for FFT

            window = 480

            synthetic_start_time = 0
            data_start_time = UTCDateTime('2025-07-29T23:24:50') # Start time
            data_end_time = UTCDateTime('2025-08-6T23:24:50')    # End time

            length = 360        # TODO: What is length?
            chan = "LHZ"        # Channel
            loc = "00"          # Location

            stream_index = 0

            data = Observation_Process(length, net, sta, chan, loc, data_start_time, data_end_time)

            xs = []
            ys = []

            power1, frequency1 = synthetic.create_spectrum(min_frequency / 1000, max_frequency / 1000, window, synthetic_start_time)
            power2, frequency2 = data.create_spectrum(min_frequency, max_frequency, data_start_time, data_start_time + 3600 * window, stream_index)

            if len(frequencies) == 0:
                frequencies.append([frequency1, frequency2])
            data_power.append(power2)
            synthetic_power.append(power1)
            networks.append(net)
            stations.append(sta)

            pbar.update(1)
    except:
        pbar.update(1)
        continue

print(len(frequencies[0]), len(power2))
# Save Data Power
df_data = pd.DataFrame(list(zip(*data_power)), columns=stations)
df_data.insert(0, "Frequency", frequencies[0][1])
df_data.to_csv('data_power.csv', index=False)

# Save Synthetic Power
df_synthetic = pd.DataFrame(list(zip(*synthetic_power)), columns=stations)
df_synthetic.insert(0, "Frequency", frequencies[0][0])
df_synthetic.to_csv('synthetic_power.csv', index=False)
