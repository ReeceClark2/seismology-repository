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
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation


mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=14)


class Compute_Normal_Mode_Spectra():
    def __init__(self, length, max_frequency, min_frequency, window_length, net, sta, chan, loc, start_time, stream_index):
        client = Client('IRIS')

        self.length = length # TODO: What is length for/mean?
        self.max_frequency, self.min_frequency = max_frequency, min_frequency
        self.window_length = window_length # Window of time to probe in hours

        # Define network, station, location, and channel
        self.net, self.sta, self.chan, self.loc = net, sta, chan, loc
        
        self.start_time = start_time
        self.end_time = self.start_time + self.window_length * 60 * 60

        self.inventory = client.get_stations(network=net, station=sta, channel=chan, location=loc, starttime=self.start_time, endtime=self.end_time, level='response')
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

        power = np.fft.fft(trace.data, n=NFFT, norm='backward')[0:NFFT] * trace.stats.delta
        frequency = np.fft.fftfreq(n=NFFT, d = trace.stats.sampling_rate)[0:NFFT] * 1000

        inventory_response = self.inventory.get_response(trace.id, trace.stats.starttime)
        response, _ = inventory_response.get_evalresp_response(trace.stats.delta, NFFT * 2, 'ACC')
        response = response[1:]

        power *= np.conjugate(response)/np.abs(response)**2

        power = power[(frequency >= min_frequency) & (frequency <= max_frequency)]
        frequency = frequency[(frequency >= min_frequency) & (frequency <= max_frequency)]

        return power, frequency
    

    def plot(self, xs, ys, labels, type):
        if type == "observed":
            for [ind, x], y in zip(enumerate(xs), ys):
                plt.plot(x, np.abs(y), alpha=0.5, label=f'{labels[ind]}')

            plt.title(f"Observed Spectrum at {sta}, {loc}")
            plt.xlabel("Frequency (mHz)")
            plt.ylabel("Power")

            plt.xlim((self.min_frequency, self.max_frequency))
            plt.legend()

            plt.show()
        
        elif type == "synthetic":
            pass


if __name__ == "__main__":
    length = 360 # TODO: What is length?

    min_frequency = 0.2 # Minimum frequency for FFT
    max_frequency = 1.2 # Maximum frequency for FFT

    net = "IU" # Network
    sta = "HRV" # Station
    chan = "LHZ" # Channel
    loc = "00" # Location
    
    stream_index = 0 # Stream index
    start_time = UTCDateTime('2025-07-29T23:24:50') # Start time

    window_length = 40 # Window length in hours

    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
    ax.set_xlabel('Frequency (mHz)')
    ax.set_ylabel('Power')
    frames = []

    df_modes = pd.read_csv("normal_modes.csv")
    mask = (df_modes["f PREM"] >= min_frequency) & (df_modes["f PREM"] <= max_frequency)
    filtered_modes = df_modes[mask]

    pbar = tqdm.tqdm(total=120)
    for i in range(120):
        current_start_time = start_time + (i * 3600)

        sta = "HRV"
        spectra = Compute_Normal_Mode_Spectra(length, max_frequency, min_frequency, window_length, net, sta, chan, loc, current_start_time, stream_index)
        power, frequency = spectra.process_data()
        
        line1, = ax.plot(frequency, np.abs(power), color='skyblue')

        title = ax.text(0.5, 1.05, f"Normal Mode Spectra at {current_start_time}", transform=ax.transAxes, va="center", ha="center")
        
        frame_artists = [line1, title]
        
        for _, row in filtered_modes.iterrows():
            f_val = row["f PREM"]
            f_label = row.iloc[0]
            
            vl = ax.axvline(f_val, color='black', linestyle=':', scaley=False)
            txt = ax.text(f_val, 0.95, f_label, transform=ax.get_xaxis_transform(), rotation=90, va='top', ha='right', fontsize=8)
            
            frame_artists.extend([vl, txt])

        frames.append(frame_artists)
        pbar.update(1)
    pbar.close()

    ani = ArtistAnimation(fig, frames, interval=250, blit=False)
    ani.save('spectra_isolated_damped_mode.mp4', writer='ffmpeg')
    plt.show()