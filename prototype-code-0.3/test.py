import os

# 1. Force math backends (like NumPy/OpenBLAS) to single-thread 
#    to prevent them from deadlocking inside worker processes.
#    MUST be set before importing numpy!
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import multiprocessing
import numpy as np
from dracula import Dracula
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 2. Force fresh process creation instead of copying memory.
    #    This prevents C-level segfaults in NumPy/SciPy during fork.
    multiprocessing.set_start_method('spawn', force=True)
    
    print("Generating data...")
    rng = np.random.default_rng()

    t = np.linspace(0, 1_000, 30_000)
    e = rng.uniform(low=-1, high=1, size=30_000)
    d = (2 * np.sin(2 * np.pi * 4 * t) * np.e ** (-0.010 * t) + 
         3 * np.sin(2 * np.pi * 5 * t) * np.e ** (-0.005 * t) + 
         4 * np.sin(2 * np.pi * 8.95 * t) * np.e ** (-0.020 * t) +
         3 * np.sin(2 * np.pi * 9.05 * t) * np.e ** (-0.025 * t) + e)

    plt.plot(t, d)
    plt.show()

    fs = [4.3, 5.1, 8.7, 9.4, 9.4]
    ks = [0.03, 0.01, 0.015, 0.027, 0.0325]

    print("Initializing Dracula...")
    model = Dracula(t, d, fs, ks)
    
    try:
        print("Dispatching workers...")
        # Note: If this still crashes, modify your Dracula.dispatch() method 
        # to include the 'sequential_debug=True' fallback loop I showed you 
        # previously to catch the exact error.
        results = model.dispatch(
            workers=5, 
            min_signals=3, 
            max_signals=5, 
            f_bw=4, 
            k_bw=0.02
        )
        print("Run complete!")
        print(results)
        
    except Exception as e:
        import traceback
        print("Caught a fatal error:")
        traceback.print_exc()