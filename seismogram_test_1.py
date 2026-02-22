#!/usr/bin/env python
from obspy.core import read
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import math
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 
import sys

import matplotlib as mpl


mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=14)


class Compute_Normal_Mode_Spectra():
    def __init__(self, length, max_frequency, min_frequency, window_length, net, sta, chan, loc, start_time, stream_index):
        client = Client('IRIS')

        self.length = length # TODO: What is length for/mean?
        self.max_frequency, self.min_frequency = max_frequency, min_frequency # TODO: What is max_frequency & min_frequency for/mean? 
        self.window_length = window_length # Window of time to probe in hours

        # Define network, station, location, and channel
        self.net, self.sta, self.chan, self.loc = net, sta, chan, loc

        # Alternate system configurations
        # net, sta, loc, chan = 'IU', 'SSPA', '00', 'LHZ'
        # net, sta, loc, chan = 'IU', 'ANMO', '00', 'LHZ'
        
        self.start_time = start_time
        self.end_time = self.start_time + self.window_length * 60 * 60

        # TODO: What does the following line do? Check obspy readthedocs.
        self.inv = client.get_stations(network=net, sta=sta, channel=chan, location=loc, starttime=self.start_time, endtime=self.end_time, level='response')
        self.stream = client.get_waveforms(network=net, station=sta, location=loc, channel=chan, starttime=self.start_time, endtime=self.end_time)
        self.trace = self.stream[stream_index]

        pass


    def read_synthetic(file):
        pass


    def process_synthetic(file):
        pass


    def load_synthetic_data(file):
        pass


    def process_data(self):
        trace = self.trace.copy()

        trace.detrend('constant')
        trace.data *= signal.get_window(('kaiser', 2. * np.pi), trace.stats.npts)
        
        NFFT = 2 ** (math.ceil(math.log(trace.stats.npts, 2)))

        power = np.fft.fft(trace.data, n=NFFT, norm='backward')[0:NFFT]*trace.stats.delta
        frequency = np.fft.fftfreq(n=NFFT, d = trace.stats.sampling_rate)[0:NFFT]*1000

        inv_resp = self.inv.get_response(trace.id, trace.stats.starttime)
        resp, _ = inv_resp.get_evalresp_response(trace.stats.delta, NFFT*2, 'ACC')
        resp = resp[1:]

        power *= np.conjugate(resp)/np.abs(resp)**2

        power = power[(frequency >= min_frequency) & (frequency <= max_frequency)]
        frequency = frequency[(frequency >= min_frequency) & (frequency <= max_frequency)]

        return power, frequency
    

    def plot(self, x, y, type):
        if type == "observed":
            plt.plot(x, np.abs(y), alpha=0.5, label='Data')

            plt.title(f"Observed Spectrum at {sta}, {loc}")
            plt.xlabel("Frequency (mHz)")
            plt.ylabel("Power")

            plt.xlim((self.min_frequency, self.max_frequency))
            plt.legend()

            plt.show()
        
        elif type == "synthetic":
            pass


if __name__ == "__main__":
    length = 360
    max_frequency, min_frequency = 2, 0.2
    window_length = 60
    net, sta, chan, loc = 'IU', 'HRV', 'LHZ', '00'
    start_time = UTCDateTime('2025-07-29T23:24:50')
    stream_index = 0

    spectra = Compute_Normal_Mode_Spectra(length, max_frequency, min_frequency, window_length, net, sta, chan, loc, start_time, stream_index)
    power, frequency = spectra.process_data()
    spectra.plot(frequency, power, "observed")
