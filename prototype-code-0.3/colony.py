import concurrent.futures

import math

from bats import BATS


def run_bats_worker(t, d, f_init, k_init, f_bw, k_bw, W, S, seed, statistics):
    model = BATS(t, d, f_init, k_init)
    return model.run_NUTS(f_bw, k_bw, W, S, seed, statistics)


class Colony():
    def __init__(self, 
                 t, 
                 d, 
                 f_init, 
                 k_init
                 ):
        
        self.t = t
        self.d = d

        self.f_init = f_init
        self.k_init = k_init


    def dispatch(self, 
                 workers,
                 f_bw=1e-3,
                 k_bw=1e-5,
                 W=1_000,
                 S=2_000,
                 statistics=True
                 ):
        
        chunk_size = math.ceil(len(self.f_init) / workers)

        tasks = []
        for i in range(workers):
            start = i * chunk_size
            end = start + chunk_size
            
            # Slice the arrays properly from 'start' to 'end'
            f_chunk = self.f_init[start:end]
            k_chunk = self.k_init[start:end]
            
            # Prevent appending empty chunks if processes > len(f_init)
            if len(f_chunk) > 0:
                tasks.append((self.t, self.d, f_chunk, k_chunk, f_bw, k_bw, W, S, i, statistics))

        worker_results = []

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_bats_worker, *task) for task in tasks]

            for future in concurrent.futures.as_completed(futures):
                try:
                    fs, ks = future.result()
                    worker_results.append((fs, ks))
                except Exception as e:
                    print(f"A BATS process failed with error: {e}")

        results = [list(group) for group in zip(*worker_results)]

        return results
