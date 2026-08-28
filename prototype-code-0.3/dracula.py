import concurrent.futures
import os
from colony import Colony, run_bats_worker
import jax.numpy as jnp
from bats import statistics
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)


class Dracula():
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
                 f_per_worker, 
                 min_signals, 
                 max_signals, 
                 f_bw=None, 
                 k_bw=None, 
                 W=1_000, 
                 S=2_000, 
                 calc_stats=True, 
                 max_cores=None
                 ):

        if max_cores is None:
            max_cores = max(1, (os.cpu_count() or 4) - 2)
            
        print(f"Limiting execution to {max_cores} concurrent workers.")

        grouped_results = {s: [] for s in range(min_signals, max_signals + 1)}
        tasks_remaining = {s: 0 for s in range(min_signals, max_signals + 1)}
        final_results = {s: None for s in range(min_signals, max_signals + 1)}

        future_metadata = {}  # Tracks what each future is doing: ('bats', signals) or ('stats', signals)
        pending_futures = set()

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
            
            # 1. Submit all initial BATS tasks
            for signals in range(min_signals, max_signals + 1):
                colony = Colony(self.t, self.d, self.f_init[:signals], self.k_init[:signals])
                colony_tasks = colony.get_tasks(f_per_worker, f_bw, k_bw, W, S)
                
                tasks_remaining[signals] = len(colony_tasks)
                
                # Edge case: If no tasks for this signal, skip
                if not colony_tasks:
                    continue

                for task_args in colony_tasks:
                    future = executor.submit(run_bats_worker, *task_args)
                    future_metadata[future] = ('bats', signals)
                    pending_futures.add(future)

            # 2. Process futures as they complete and dynamically add stats tasks
            while pending_futures:
                # Wait for at least one future to finish
                done, pending_futures = concurrent.futures.wait(
                    pending_futures, 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                for future in done:
                    # Pop metadata so we don't leak memory
                    task_type, signals = future_metadata.pop(future)
                    
                    try:
                        if task_type == 'bats':
                            fs, ks = future.result()
                            grouped_results[signals].append((fs, ks))
                            tasks_remaining[signals] -= 1
                            
                            # If all BATS tasks for this specific signal are done, launch statistics
                            if tasks_remaining[signals] == 0:
                                colony_res = grouped_results[signals]
                                
                                # Zip to transpose [(fs1, ks1), (fs2, ks2)] into [(fs1, fs2), (ks1, ks2)]
                                combined_fs_chunks, combined_ks_chunks = [list(group) for group in zip(*colony_res)]
                                
                                # Concatenate the chunks into flat 1D arrays
                                flat_fs = jnp.concatenate(combined_fs_chunks)
                                flat_ks = jnp.concatenate(combined_ks_chunks)
                                
                                # Launch the new statistics task
                                stats_future = executor.submit(
                                    statistics, 
                                    self.t, 
                                    self.d, 
                                    flat_fs,  # Now a single 1D array of length 'signals'
                                    flat_ks   # Now a single 1D array of length 'signals'
                                )
                                
                                # Tag it and add it to our active pool
                                future_metadata[stats_future] = ('stats', signals)
                                pending_futures.add(stats_future)
                                
                        elif task_type == 'stats':
                            # Store the final calculated statistics
                            stats_result = future.result()
                            final_results[signals] = stats_result
                            
                    except Exception as e:
                        print(f"Task '{task_type}' for {signals} signals failed with error: {e}")

        # Return results ordered by signal count
        return [final_results[s] for s in range(min_signals, max_signals + 1)]
    