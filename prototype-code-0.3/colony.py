import math
from bats import BATS


def run_bats_worker(t, d, f_init, k_init, f_bw, k_bw, W, S, seed, statistics):
    model = BATS(t, d, f_init, k_init)
    return model.run_NUTS(f_bw, k_bw, W, S, seed, statistics)


class Colony():
    def __init__(self, t, d, f_init, k_init):
        self.t = t
        self.d = d

        parameters = sorted(zip(f_init, k_init))

        self.f_init, self.k_init = map(list, zip(*parameters))


    def get_tasks(self, workers, f_bw=1e-3, k_bw=1e-5, W=1_000, S=2_000, statistics=True):
        """Generates the parameters for each worker task without executing them."""
        chunk_size = math.ceil(len(self.f_init) / workers)
        tasks = []
        
        for i in range(workers):
            start = i * chunk_size
            end = start + chunk_size
            
            f_chunk = self.f_init[start:end]
            k_chunk = self.k_init[start:end]
            
            if len(f_chunk) > 0:
                tasks.append((self.t, self.d, f_chunk, k_chunk, f_bw, k_bw, W, S, i, statistics))
                
        return tasks