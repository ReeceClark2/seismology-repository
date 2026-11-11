import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, sosfiltfilt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to the two-column ASCII file")
    parser.add_argument("f_min", type=float, help="Minimum bandpass frequency")
    parser.add_argument("f_max", type=float, help="Maximum bandpass frequency")
    args = parser.parse_args()

    data = np.loadtxt(args.file)
    t, d = data[:, 0], data[:, 1]

    dt = np.mean(np.diff(t))
    fs = 1.0 / dt

    sos = butter(
        4,
        [args.f_min, args.f_max],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    filtered = sosfiltfilt(sos, d)

    freq = np.fft.rfftfreq(len(d), d=dt)
    raw_amplitude = np.abs(np.fft.rfft(d))
    filtered_amplitude = np.abs(np.fft.rfft(filtered))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(t, d, linewidth=0.8)
    axes[0, 0].set(title="Original time series", xlabel="Time", ylabel="Amplitude")

    axes[0, 1].plot(t, filtered, linewidth=0.8)
    axes[0, 1].set(title="Bandpass-filtered time series", xlabel="Time", ylabel="Amplitude")

    axes[1, 0].plot(freq, raw_amplitude, linewidth=0.8)
    axes[1, 0].set(title="Original amplitude spectrum", xlabel="Frequency", ylabel="Amplitude")

    axes[1, 1].plot(freq, filtered_amplitude, linewidth=0.8)
    axes[1, 1].set(title="Filtered amplitude spectrum", xlabel="Frequency", ylabel="Amplitude")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()