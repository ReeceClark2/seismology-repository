import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, sosfreqz
from pathlib import Path

def filter_data(t, d, f_min, f_max, order=8):
    f_min = 0.00028
    f_max = 0.00500
    fs = 1 / np.mean(np.diff(t))
    sos = butter(order, [f_min, f_max], btype="bandpass", fs=fs, output="sos")
    filtered_d = sosfiltfilt(sos, d)
    trim = int(5 * 3600 * fs)
    t_trimmed = t[trim:-trim]
    return t_trimmed - t_trimmed[0], filtered_d[trim:-trim]

def plot_filter_response(t, f_min, f_max, order=48):
    fs = 1 / np.mean(np.diff(t))
    sos = butter(order, [f_min, f_max], btype="bandpass", fs=fs, output="sos")
    f = np.linspace(0.0001, 0.005, 5000)
    _, h = sosfreqz(sos, worN=f, fs=fs)
    amplitude = np.abs(h) ** 2
    db = 20 * np.log10(np.maximum(amplitude, 1e-12))
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(f, amplitude)
    ax[0].set_ylabel("Amplitude ratio")
    ax[0].set_title(f"Forward-backward filter response ({f_min}–{f_max} Hz passband)")
    ax[0].grid(True)
    ax[1].plot(f, db)
    ax[1].set_xlabel("Frequency (Hz)")
    ax[1].set_ylabel("Magnitude (dB)")
    ax[1].set_xlim(0.0001, 0.005)
    ax[1].grid(True)
    plt.tight_layout()
    plt.show()

def load_data(type, event, network, station):
    if type == "observed":
        path = Path.cwd() / type / event / f"{network}_{station}_TS.ascii"
    elif type == "synthetic":
        path = Path.cwd() / type / event / f"{network}_{station}.dat"
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]

if __name__ == "__main__":
    type = "synthetic"
    event = "tohoku"
    network = "II"
    station = "BFO"

    t, d = load_data(type, event, network, station)
    plot_filter_response(t, 0.00028, 0.002)