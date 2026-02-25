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
import csv

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=14)


class Synthetic_Process():
    def __init__(self, filename, type):
        if type == 0:       # Compute synthetic time series
            self.data = self.parse_file(filename)
            self.times = self.data.iloc[:, 0]
            self.intensities = self.data.iloc[:, 1]

        elif type == 1:     # Compute synthetic spectrum
            pass


    def create_spectrum(self, minimum_frequency, maximum_frequency, start_time, end_time, stream_index):
        intensities = self.intensities.copy()

        # Use windowing function for DFT
        intensities *= signal.get_window(('kaiser', 2. * np.pi), len(intensities))
        
        NFFT = 2 ** (math.ceil(math.log(len(intensities), 2)))

        power = np.fft.fft(intensities, n=NFFT, norm='backward')[0:NFFT] * trace.stats.delta
        frequency = np.fft.fftfreq(n=NFFT, d = trace.stats.sampling_rate)[0:NFFT] * 1000

        inventory_response = self.inventory.get_response(trace.id, trace.stats.starttime)
        response, _ = inventory_response.get_evalresp_response(trace.stats.delta, NFFT * 2, 'ACC')
        response = response[1:]

        power *= np.conjugate(response) / np.abs(response)**2

        # Filter power and frequency to parameters
        power = power[(frequency >= minimum_frequency) & (frequency <= maximum_frequency)]
        frequency = frequency[(frequency >= minimum_frequency) & (frequency <= maximum_frequency)]

        return power, frequency
    

    def parse_file(self, filename):
        dataframe = pd.read_csv(filename, sep=' ', encoding='ascii', header=None)

        return dataframe    


if __name__=="__main__":
    file = "timeseries_Russia/IU_HRV_TS.ascii"

    Synthetic_Process(file, 0)
