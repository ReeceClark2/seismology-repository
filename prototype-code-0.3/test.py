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
    rng = np.random.default_rng(42) # Seeded for reproducibility

    t = np.linspace(0, 1_000, 30_000)
    e = rng.uniform(low=-1, high=1, size=30_000)
    
    # ---------------------------------------------------------
    # Generate 20 true signals dynamically 
    # ---------------------------------------------------------
    d = np.zeros_like(t)
    
    # Generate 20 underlying frequencies, decay rates, and amplitudes
    true_fs = np.linspace(2, 21, 20)  # Frequencies spread from 2 to 21
    true_ks = rng.uniform(0.005, 0.04, size=20) # Random decay rates
    true_amps = rng.uniform(1, 5, size=20) # Random amplitudes

    # Build the synthetic data waveform containing 20 signals
    for A, f, k in zip(true_amps, true_fs, true_ks):
        d += A * np.sin(2 * np.pi * f * t) * np.exp(-k * t)
        
    d += e # Add the noise vector

    plt.plot(t, d)
    plt.title("20-Signal Synthetic Data")
    plt.show()

    # ---------------------------------------------------------
    # Generate 25 rough "guesses" for Dracula
    # ---------------------------------------------------------
    # We take the 20 true parameters, add some noise/error to them to 
    # simulate "rough" guesses, and then append 5 completely fake guesses.
    
    fs = [f + rng.uniform(-0.4, 0.4) for f in true_fs] + [23.1, 24.5, 26.0, 27.2, 28.5]
    
    # Ensure decay rates don't drop below 0.001 after adding noise
    ks = [max(0.001, k + rng.uniform(-0.01, 0.01)) for k in true_ks] + [0.01, 0.02, 0.015, 0.03, 0.025]

    print("Initializing Dracula...")
    model = Dracula(t, d, fs, ks)
    
    try:
        print("Dispatching workers...")
        # Updated sample range: 15 to 25
        results = model.dispatch(
            workers=5, 
            min_signals=15, 
            max_signals=25, 
            f_bw=4, 
            k_bw=0.02
        )
        print("Run complete!")
        print(results)
        
    except Exception as e:
        import traceback
        print("Caught a fatal error:")
        traceback.print_exc()