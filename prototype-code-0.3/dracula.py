import concurrent.futures

import math
import os

from colony import Colony


def run_colony_worker(t, d, f_init, k_init, workers, f_bw, k_bw, W, S, statistics):
    colony = Colony(t, d, f_init, k_init)
    return colony.dispatch(workers, f_bw, k_bw, W, S, statistics)


class Dracula():
    def __init__(self, 
                 t, 
                 d, 
                 f_init, 
                 k_init,
                 sort_signals=True
                 ):
        
        self.t = t
        self.d = d

        self.f_init = f_init
        self.k_init = k_init


    def dispatch(self, 
                 workers,
                 min_signals,
                 max_signals,
                 f_bw=1e-3,
                 k_bw=1e-5,
                 W=1_000,
                 S=2_000,
                 statistics=True
                 ):

        tasks = []

        for signals in range(min_signals, max_signals + 1):
            tasks.append((self.t, self.d, self.f_init[:signals], self.k_init[:signals], workers, f_bw, k_bw, W, S, statistics))

        worker_results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_signals-min_signals+1) as executor:
            futures = [executor.submit(run_colony_worker, *task) for task in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    fs, ks = future.result()
                    worker_results.append((fs, ks))
                except Exception as e:
                    print(f"A colony process failed with error: {e}")

        print(worker_results)

