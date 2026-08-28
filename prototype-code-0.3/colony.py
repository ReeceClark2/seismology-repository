import math
from bats import BATS
from scipy.signal import butter, sosfilt
import matplotlib.pyplot as plt


def run_bats_worker(t, d, f_init, k_init, f_bw, k_bw, W, S, seed):
    model = BATS(t, d, f_init, k_init)
    return model.run_NUTS(f_bw, k_bw, W, S, seed)

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Use second-order sections for numerical stability
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y

class Colony():
    def __init__(self, t, d, f_init, k_init):
        self.t = t
        self.d = d

        parameters = sorted(zip(f_init, k_init))

        self.f_init, self.k_init = map(list, zip(*parameters))


    def get_tasks(self, f_per_worker, f_bw=1e-3, k_bw=1e-5, W=1_000, S=2_000):
        workers = math.ceil(len(self.f_init) / f_per_worker)
        chunk_size = math.ceil(f_per_worker)
        tasks = []

        for i in range(workers):
            start = i * chunk_size
            end = start + chunk_size

            if end > len(self.f_init):
                end = len(self.f_init)

            filtered_d = butter_bandpass_filter(self.d, self.f_init[start] - (5 * f_bw[start]), self.f_init[end-1] + (5 * f_bw[end-1]), 1/25, order=6)

            f_chunk = self.f_init[start:end]
            k_chunk = self.k_init[start:end]
            f_bw_chunk = f_bw[start:end]
            k_bw_chunk = k_bw[start:end]
            
            if len(f_chunk) > 0:
                tasks.append((self.t, filtered_d, f_chunk, k_chunk, f_bw_chunk, k_bw_chunk, W, S, i))
                
        return tasks
    