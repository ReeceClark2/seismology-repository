#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import signal
import math

# Formatting for publication-quality plots
mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=True)
mpl.rc('font', size=14)

class Synthetic_Process():
    def __init__(self, filename, type_code=0, min_freq=None, max_freq=None):
        """
        type_code: 0 for time series processing
        min_freq, max_freq: Bandpass bounds in Hz
        """
        if type_code == 0:
            self.data = self.parse_file(filename)
            # Column 0 is time in seconds, Column 1 is intensity
            self.times = self.data.iloc[:, 0].values
            self.intensities = self.data.iloc[:, 1].values
            
            # Calculate actual delta from the file data
            self.delta = np.mean(np.diff(self.times)) 
            self.fs = 1.0 / self.delta

            # Apply bandpass filter if bounds are provided
            if min_freq is not None and max_freq is not None:
                print(f"Applying Bandpass: {min_freq}Hz to {max_freq}Hz")
                self.intensities = self.apply_bandpass(min_freq, max_freq)

    def apply_bandpass(self, lowcut, highcut, order=4):
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        
        if low <= 0 or high >= 1:
            print(f"Warning: Frequencies out of Nyquist bounds. Skipping filter.")
            return self.intensities

        b, a = signal.butter(order, [low, high], btype='band')
        # Detrending prevents edge offsets from blowing up the filter
        detrended = signal.detrend(self.intensities)
        return signal.filtfilt(b, a, detrended)

    def create_spectrum(self, min_f, max_f, window_hours, start_hour=0):
        """
        Slices the data based on hours and computes the FFT.
        """
        # 1. Convert hours to sample indices
        # Index = (Hours * 3600 seconds/hour) / delta_seconds
        start_idx = int(round((start_hour * 3600) / self.delta))
        window_samples = int(round((window_hours * 3600) / self.delta))
        end_idx = start_idx + window_samples

        # 2. Extract the segment
        if end_idx > len(self.intensities):
            print(f"Warning: Window exceeds data length. Clipping to max available.")
            segment = self.intensities[start_idx:]
        else:
            segment = self.intensities[start_idx:end_idx]

        # 3. Apply Tapering (Kaiser window)
        # Tapering is vital to prevent spectral leakage at the edges of the window
        taper = signal.get_window(('kaiser', 2. * np.pi), len(segment))
        segment_weighted = segment * taper
        
        # 4. Compute FFT
        # NFFT is the next power of 2 for computational efficiency
        nfft = 2 ** (math.ceil(math.log(len(segment_weighted), 2)))
        
        # frequency in Hz
        freqs = np.fft.fftfreq(n=nfft, d=self.delta)
        # Power spectrum
        full_power = np.fft.fft(segment_weighted, n=nfft) * self.delta

        # 5. Mask for desired frequency range (Positive frequencies only)
        # mask = (freqs >= min_f) & (freqs <= max_f)
        return full_power, freqs

    def parse_file(self, filename):
        # Assumes space-separated: [Time_Seconds Intensity]
        return pd.read_csv(filename, sep=' ', header=None, encoding='ascii')

# --- Main Execution ---
if __name__ == "__main__":
    file_path = "timeseries_Russia/IU_HRV_TS.ascii" 

    # Parameters
    f_min_hz = 0.00044 # 0.2 mHz
    f_max_hz = 0.00055 # 1.2 mHz
    
    analysis_window_hours = 400 # Length of data to analyze
    start_offset_hours = 0     # Start at the beginning of the file

    # 1. Initialize and Filter
    # We pass the filter bounds here to clean the time series immediately
    data_proc = Synthetic_Process(file_path, type_code=0, 
                                  min_freq=f_min_hz, max_freq=f_max_hz)

    # 2. Generate Spectrum
    # Note: We pass Hz values directly here
    power, freq = data_proc.create_spectrum(f_min_hz, f_max_hz, 
                                            analysis_window_hours, 
                                            start_offset_hours)

    # 3. Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(freq * 1000, np.abs(power), color='black', lw=1) # Plot in mHz for readability
    plt.xlabel(r"Frequency (mHz)")
    plt.ylabel(r"Amplitude $|F(\omega)|$")
    plt.title(f"Spectrum Analysis: {analysis_window_hours} Hour Window")
    plt.grid(alpha=0.3)
    plt.show()