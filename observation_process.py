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


mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=14)


class Observation_Process():
    def __init__(self, length, net, sta, chan, loc, start_time, end_time):
        client = Client('IRIS')

        self.length = length # TODO: What is length for/mean?

        # Define network, station, location, and channel
        self.net, self.sta, self.chan, self.loc = net, sta, chan, loc
        
        self.start_time = start_time
        self.end_time = end_time

        self.inventory = client.get_stations(network=net, station=sta, channel=chan, location=loc, starttime=self.start_time, endtime=self.end_time, level='response')
        self.stream = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=self.start_time, endtime=self.end_time)


    def create_spectrum(self, minimum_frequency, maximum_frequency, start_time, end_time, stream_index):
        stream = self.stream.copy()
        stream.trim(start_time, end_time)
        trace = stream[stream_index]

        # Detrend data (primarily to filter tides)
        trace.detrend('constant')

        # Use windowing function for DFT
        trace.data *= signal.get_window(('kaiser', 2. * np.pi), trace.stats.npts)
        
        NFFT = 2 ** (math.ceil(math.log(trace.stats.npts, 2)))

        power = np.fft.fft(trace.data, n=NFFT, norm='backward')[0:NFFT] * trace.stats.delta
        frequency = np.fft.fftfreq(n=NFFT, d = trace.stats.sampling_rate)[0:NFFT] * 1000

        inventory_response = self.inventory.get_response(trace.id, trace.stats.starttime)
        response, _ = inventory_response.get_evalresp_response(trace.stats.delta, NFFT * 2, 'ACC')
        response = response[1:]
        print(trace.stats.delta, trace.stats.sampling_rate, trace.stats.npts)
        power *= np.conjugate(response) / np.abs(response)**2

        # Filter power and frequency to parameters
        power = power[(frequency >= minimum_frequency) & (frequency <= maximum_frequency)]
        frequency = frequency[(frequency >= minimum_frequency) & (frequency <= maximum_frequency)]

        return power, frequency


if __name__ == "__main__":
    length = 360        # TODO: What is length?

    min_frequency = 0.2 # Minimum frequency for FFT
    max_frequency = 1.2 # Maximum frequency for FFT

    net = "IU"          # Network
    sta = "HRV"         # Station
    chan = "LHZ"        # Channel
    loc = "00"          # Location
    
    stream_index = 0                                # Stream index
    start_time = UTCDateTime('2025-07-29T23:24:50') # Start time
    end_time = UTCDateTime('2025-08-4T23:24:50')    # End time

    data = Observation_Process(length, net, sta, chan, loc, start_time, end_time)

    end_time = UTCDateTime('2025-07-30T13:24:50')
    power, frequency = data.create_spectrum(min_frequency, max_frequency, start_time, end_time, stream_index)
