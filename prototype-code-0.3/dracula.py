import concurrent.futures
import os
from colony import Colony, run_bats_worker


class Dracula():
    def __init__(self, t, d, f_init, k_init, sort_signals=True):
        self.t = t
        self.d = d
        self.f_init = f_init
        self.k_init = k_init


    def dispatch(self, workers, min_signals, max_signals, f_bw=1e-3, k_bw=1e-5, W=1_000, S=2_000, statistics=True, max_cores=None):

        if max_cores is None:
            max_cores = max(1, (os.cpu_count() or 4) - 2)
            
        print(f"Limiting execution to {max_cores} concurrent BATS instances.")

        all_tasks = []
        
        for signals in range(min_signals, max_signals + 1):
            colony = Colony(self.t, self.d, self.f_init[:signals], self.k_init[:signals])
            colony_tasks = colony.get_tasks(workers, f_bw, k_bw, W, S, statistics)
            
            for task in colony_tasks:
                # Tag each task with the number of 'signals' so we can group the results later
                all_tasks.append((signals, task))

        grouped_results = {s: [] for s in range(min_signals, max_signals + 1)}

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            
            # Map futures to their original signal count
            future_to_signal = {
                executor.submit(run_bats_worker, *task_args): signals 
                for signals, task_args in all_tasks
            }

            for future in concurrent.futures.as_completed(future_to_signal):
                signals = future_to_signal[future]
                try:
                    fs, ks = future.result()
                    grouped_results[signals].append((fs, ks))
                except Exception as e:
                    print(f"A BATS process for {signals} signals failed with error: {e}")

        final_results = []
        for signals in range(min_signals, max_signals + 1):
            colony_res = grouped_results[signals]
            
            if colony_res:
                results = [list(group) for group in zip(*colony_res)]
                final_results.append(results)
            else:
                final_results.append([])

        return final_results