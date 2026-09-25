from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import chain
from dataclasses import dataclass, field, fields
from typing import Any, Optional
from pathlib import Path
from datetime import datetime
import traceback
import math
from copy import deepcopy
from collections import defaultdict
import os
import psutil

import numpy as np

os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false "
    "intra_op_parallelism_threads=1"
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import jax.numpy as jnp
from jax.typing import ArrayLike

from tqdm import tqdm

import bats
import utils
import log_utils


def initialize_worker(core_queue):
    '''
    Windows and Linux compatible functon for diagnosing core availibility.
    '''

    worker_cores = core_queue.get()

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, worker_cores)

        active_cores = sorted(os.sched_getaffinity(0))
        message = f"PID {os.getpid()} using cores {active_cores}"
    else:
        available_cores = os.cpu_count() or 1
        message = (
            f"PID {os.getpid()} on Windows; "
            f"affinity not explicitly pinned; "
            f"{available_cores} logical CPUs available"
        )

    print(message, flush=True)


@dataclass
class SignalSpace:
    '''
    Defines the set of allowed frequencies and decay rates that bound 
    the whole of signal-decay rate space.
    '''
    f_min: float
    f_max: float
    k_min: float
    k_max: float

@dataclass
class GridSearchArgs:
    '''
    Defines the resolution sought for a grid search to find initial conditions.
    '''
    f_points: int
    k_points: int

@dataclass
class NUTSArgs:
    '''
    Defines the NUTS dictionaries to pass all numpyro kwargs.
    '''
    seed: Any
    nuts_kwargs: dict[str, Any] = field(default_factory=dict)
    mcmc_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class InitialConditionsTask:
    '''
    Defines the allowed parameters of an initial conditions worker.
    '''
    t: ArrayLike
    d: ArrayLike
    signal_space: SignalSpace
    depth: int
    grid_search_args: GridSearchArgs
    nuts_args: NUTSArgs
    perform_minimize: bool
    path: str
    

def run_initial_conditions_worker(t, d, signal_space, depth, grid_search_args, nuts_args, perform_minimize, path):
    '''
    Runs an instance of the initial conditions worker to find all signals in a subband.
    '''
    path.mkdir(parents=True, exist_ok=True)

    def update_signals(
            t,
            d,
            signal_space,
            grid_search_args,
            nuts_args=None,
            perform_minimize=False,
            signals=None,
            signals_bw=None,
            model=None,
        ):
        '''
        Find the next signal and return it and its beamwidth. At minimum, must perform a grid search, 
        but can run grid search and NUTS sample to typical set and lbfgs to local maximum probability.
        '''
        if model is not None:
            d = d - model

        signal_candidate, probability_surface = bats.grid_search(
            t, 
            d,
            grid_search_args.f_points, 
            signal_space.f_min, signal_space.f_max, 
            grid_search_args.k_points, signal_space.k_min, 
            signal_space.k_max, 
            return_probability_surface=True
        )

        signal_candidate = jnp.asarray(signal_candidate).reshape(1, 2)
        signal_candidate_bw = jnp.asarray(get_signal_bw(signal_space)).reshape(1, 2)

        if signals is None:
            signals = signal_candidate
            signals_0 = signal_candidate
            signals_bw = signal_candidate_bw
        else:
            signals = jnp.asarray(signals).reshape(-1, 2)
            signals_0 = jnp.concatenate((signals, signal_candidate), axis=0)
            signals_bw = jnp.concatenate((signals_bw, signal_candidate_bw), axis=0)
            signals = signals_0
        if model is not None:
            d = d + model

        if nuts_args is not None:
            signals = bats.nuts(
                t, 
                d, 
                signal_space,
                signals, 
                signals_bw, 
                nuts_args.nuts_kwargs, 
                nuts_args.mcmc_kwargs, 
                nuts_args.run_kwargs, 
                nuts_args.seed
            )

        if perform_minimize is True:
            signals = bats.minimize(
                t, 
                d, 
                signal_space,
                signals
            )

        log_utils.plot_probability_surface(path / f"{len(signals)}_probability_surface.png", probability_surface, f"Probability Surface of Signal {len(signals)}")
        log_utils.plot_signal_space(path / f"{len(signals)}_signal_space.png", signals, f"Signal Space of {len(signals)} Signals", signals_0=signals_0, signals_bw=signals_bw, signal_space=signal_space)

        return signals, signals_bw

    # subband_bw = signal_space.f_max - signal_space.f_min

    subband_t = t.copy()
    subband_d = utils.filter(subband_t, d.copy(), signal_space.f_min, signal_space.f_max)

    t_max = jnp.log(0.2) / (-signal_space.k_min)

    mask = t < t_max

    subband_t = subband_t[mask]
    subband_d = subband_d[mask]

    t = t[mask]
    d = d[mask]

    log_utils.plot_time_series(path / "raw_time_series.png", subband_t, subband_d, "Original Time Series")
    log_utils.plot_fourier_space(path / "raw_fourier_space", t, d, "Original Fourier Space", signal_space.f_min, signal_space.f_max, 10_000)

    def get_signal_bw(signal_space):
        '''
        Small helper function to return beamwidth of signal. TODO: Make dynamic to individual signal, TODO: Allow the halfwidth to be an argument
        '''
        f_halfwidth = (signal_space.f_max - signal_space.f_min) / 8
        log_k_halfwidth = (jnp.log(signal_space.k_max) - jnp.log(signal_space.k_min)) / 8

        return (f_halfwidth, log_k_halfwidth)

    glob_lls = []
    noise_variances = []
    snrs = []

    signals, signals_bw = update_signals(
        subband_t,
        subband_d,
        signal_space,
        grid_search_args,
        nuts_args,
        perform_minimize=perform_minimize,
    )

    signals_by_depth = {}
    signals_by_depth[len(signals)] = deepcopy(signals)
    signals_bw_by_depth = {}
    signals_bw_by_depth[len(signals_bw)] = deepcopy(signals_bw)

    glob_lls.append(bats.get_glob_ll(t, d, signals))
    noise_variances.append(bats.get_noise_variance(subband_t, subband_d, signals))
    snrs.append(bats.get_snr(subband_t, subband_d, signals))

    model = bats.get_model(subband_t, subband_d, signals)
    log_utils.plot_time_series(path / f"{len(signals)}_signal_time_series.png", subband_t, subband_d, f"Time Series for {len(signals)} Signal Model", model)
    log_utils.plot_fourier_space(path / f"{len(signals)}_signal_fourier_space", t, d, f"Fourier Space for {len(signals)} Signal Model", signal_space.f_min, signal_space.f_max, 10_000, signals, model)

    reason = "depth"
    while True:
        signals, signals_bw = update_signals(
            subband_t,
            subband_d,
            signal_space,
            grid_search_args,
            nuts_args,
            perform_minimize=perform_minimize,
            signals=signals,
            signals_bw=signals_bw,
            model=model,
        )

        model = bats.get_model(subband_t, subband_d, signals)
        log_utils.plot_time_series(path / f"{len(signals)}_signal_time_series.png", subband_t, subband_d, f"Time Series for {len(signals)} Signal Model", model)
        log_utils.plot_fourier_space(path / f"{len(signals)}_signal_fourier_space", t, d, f"Fourier Space for {len(signals)} Signal Model", signal_space.f_min, signal_space.f_max, 10_000, signals, model)

        signals_by_depth[len(signals)] = deepcopy(signals)
        signals_bw_by_depth[len(signals_bw)] = deepcopy(signals_bw)
        noise_variances.append(bats.get_noise_variance(subband_t, subband_d, signals))
        snrs.append(bats.get_snr(subband_t, subband_d, signals))
        glob_lls.append(bats.get_glob_ll(t, d, signals))

        if len(signals) >= depth:
            break

    log_utils.save_subband_csv(path / "subband_results.csv", signals, noise_variances, snrs, glob_lls)

    index = glob_lls.index(max(glob_lls))
    stop = index + 1

    signals = deepcopy(signals_by_depth[stop])
    signals_bw = deepcopy(signals_bw_by_depth[stop])
    noise_variances = noise_variances[:stop]
    snrs = snrs[:stop]
    glob_lls = glob_lls[:stop]
    signals = deepcopy(signals_by_depth[len(signals)])
    signals_bw = deepcopy(signals_bw_by_depth[len(signals_bw)])

    noise_variance = bats.get_noise_variance(subband_t, subband_d, signals)
    snr = bats.get_snr(subband_t, subband_d, signals)
    glob_ll = bats.get_glob_ll(t, d, signals)

    return {
        "f_min": signal_space.f_min,
        "f_max": signal_space.f_max,
        "signals": signals,
        "signals_bw": signals_bw,
        "noise_variance": noise_variance,
        "snr": snr,
        "glob_ll": glob_ll,
        "reason": reason,
    }

def run_initial_conditions_worker_wrapper(config: InitialConditionsTask):
    '''
    Wrapper for launching initial conditions worker and flattening task dictionary.
    '''
    config_dict = {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }

    try:
        return run_initial_conditions_worker(**config_dict)

    except BaseException as exc:
        worker_traceback = traceback.format_exc()

        print(
            "\n========== WORKER TRACEBACK ==========\n"
            f"Exception type: {type(exc).__name__}\n"
            f"Exception: {exc}\n"
            f"{worker_traceback}"
            "======================================\n",
            flush=True,
        )

        raise RuntimeError(
            f"Worker failed with {type(exc).__name__}: {exc}\n\n"
            f"Original worker traceback:\n{worker_traceback}"
        ) from None

@dataclass
class SampleTask:
    '''
    Defines the allowed parameters for the sampling worker.
    '''
    t: ArrayLike
    d: ArrayLike
    signal_space: SignalSpace
    signals: Any
    signals_bw: Any
    signal_indices: list[int]
    nuts_args: NUTSArgs
    perform_minimize: bool
    path: str

def run_sample_worker(
    t,
    d,
    signal_space,
    signals,
    signals_bw,
    signal_indices,
    nuts_args,
    perform_minimize,
    path,
):
    '''
    Runs an instance of the sample worker that improves initial condtions by 
    allowing more signals to covary.
    '''    
    path.mkdir(parents=True, exist_ok=True)

    d = utils.filter(t, d, signal_space.f_min, signal_space.f_max)

    t_max = jnp.log(0.2) / (-signal_space.k_min)

    mask = t < t_max

    t = t[mask]
    d = d[mask]

    log_utils.plot_time_series(path / "raw_time_series.png", t, d, "Original Time Series")
    log_utils.plot_fourier_space(path / "raw_fourier_space", t, d, "Original Fourier Space", signal_space.f_min, signal_space.f_max, 10_000)

    signals_0 = signals.copy()
    signals = bats.nuts(
        t,
        d,
        signal_space,
        signals,
        signals_bw,
        nuts_args.nuts_kwargs,
        nuts_args.mcmc_kwargs,
        nuts_args.run_kwargs,
        nuts_args.seed
    )
    if perform_minimize is True:
        signals = bats.minimize(
            t,
            d,
            signal_space,
            signals
        )

    log_utils.plot_signal_space(path / f"{len(signals)}_signal_space.png", signals, f"Signal Space of {len(signals)} Signals", signals_0=signals_0, signals_bw=signals_bw, signal_space=signal_space)

    model = bats.get_model(t, d, signals)
    log_utils.plot_time_series(path / f"model_time_series.png", t, d, "Model Time Series", model)
    log_utils.plot_fourier_space(path / f"{len(signals)}_signal_fourier_space", t, d, f"Fourier Space for {len(signals)} Signal Model", signal_space.f_min, signal_space.f_max, 10_000, signals, model)
    log_utils.save_block_csv(path / "block_results.csv", signals)

    return list(zip(signal_indices, signals))


def run_sample_worker_wrapper(config: SampleTask):
    '''
    Wrapper function for the sample worker to flatten and pass SampleTask dictionary.
    '''
    config_dict = {
        field.name: getattr(config, field.name)
        for field in fields(config)
    }

    try:
        return run_sample_worker(**config_dict)

    except BaseException as exc:
        worker_traceback = traceback.format_exc()

        print(
            "\n========== WORKER TRACEBACK ==========\n"
            f"Exception type: {type(exc).__name__}\n"
            f"Exception: {exc}\n"
            f"{worker_traceback}"
            "======================================\n",
            flush=True,
        )

        raise RuntimeError(
            f"Worker failed with {type(exc).__name__}: {exc}\n\n"
            f"Original worker traceback:\n{worker_traceback}"
        ) from None


def create_deliverables(path, t, d, signals, signal_space):
    model = bats.get_model(t, d, signals)
    amplitudes = bats.get_amplitudes(t, d, signals)
    uncertainties = bats.get_uncertainties(t, d, signals)

    noise_variance = bats.get_noise_variance(t, d, signals)
    snr = bats.get_snr(t, d, signals)

    log_utils.plot_time_series(
        path / f"{len(signals)}_time_series.png", 
        t, 
        d, 
        f"Time Series ({len(signals)} signal model)", 
        model
    )
    log_utils.plot_fourier_space(
        path / f"{len(signals)}_fourier_space.png", 
        t,
        d,
        f"Fourier Space ({len(signals)} signal model)", 
        signal_space.f_min,
        signal_space.f_max,
        10_000,
        signals,
        model
    )
    log_utils.plot_signal_space(
        path / f"{len(signals)}_signal_space.png", 
        signals,
        f"Signal Space ({len(signals)} signal model)", 
        uncertainties=uncertainties,
        signal_space=signal_space
    )

    log_utils.save_signals_csv(
        path / "signals.csv",
        signals,
        amplitudes,
        uncertainties
    )
    log_utils.save_report_txt(
        path / "report.txt",
        len(signals),
        noise_variance,
        snr
    )



class Dracula():
    def __init__(
            self, 
            t: ArrayLike, 
            d: ArrayLike,
            f_min: float,
            f_max: float,
            k_min: float,
            k_max: float,
            max_workers: int=1,
            path: str=None
        ):
        '''
        Model initialization for the time series, signal space, maximum workers
        allowed, and directory to store data.
        '''

        self.t = jnp.asarray(t)
        self.d = jnp.asarray(d)

        self.signal_space = SignalSpace(
            f_min=f_min,
            f_max=f_max,
            k_min=k_min,
            k_max=k_max
        )

        self.max_workers = max_workers

        if path is None:
            path = Path.cwd() / "dracula"
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.path = path.parent / f"{path.name}_{timestamp}"
        else:
            path = Path.cwd() / path
            self.path = path.parent / path.name
        
        self.path.mkdir(parents=True, exist_ok=True)

        self.default_grid_search_args = GridSearchArgs(
            f_points=100,
            k_points=100
        )

        nuts_kwargs = dict()
        mcmc_kwargs = dict(
            num_warmup=500,
            num_samples=2_000,
            num_chains=1
        )
        run_kwargs = dict()
        self.default_nuts_args_sample = NUTSArgs(
            nuts_kwargs=nuts_kwargs,
            mcmc_kwargs=mcmc_kwargs,
            run_kwargs=run_kwargs,
            seed=42
        )

    def initialize(
            self,
            grid_search_args: GridSearchArgs,
            nuts_args: NUTSArgs,
            subband_count: int = 5,
            subband_scaling_factor: float = 1,
            depth: int = 5,
            cores_per_worker: int = 1,
    ):
        print("Finding initial conditions...")

        path = self.path / "initialize"
        path.mkdir(parents=True, exist_ok=True)
        pbar = tqdm(total=subband_count)
        
        if subband_count == 1:
            subbands = [(self.signal_space.f_min, self.signal_space.f_max)]

        else:
            bandwidth = self.signal_space.f_max - self.signal_space.f_min
            subbands = np.empty((subband_count), dtype=object)
            r = subband_scaling_factor ** (1 / (subband_count - 1))

            if subband_scaling_factor == 1:
                subband_width_0 = bandwidth / subband_count
            else:
                subband_width_0 = bandwidth * ((1 - r) / (1 - r ** subband_count))

            subband_min = self.signal_space.f_min
            for i in range(subband_count):
                subband_width = subband_width_0 * (r ** i)
                subband_max = subband_min + subband_width

                subbands[i] = (subband_min, subband_max)

                subband_min = subband_max

        tasks: list[InitialConditionsTask | None] = [None] * len(subbands)
        for ind, _ in enumerate(subbands):
            signal_space = SignalSpace(
                f_min=subbands[ind][0],
                f_max=subbands[ind][1],
                k_min=self.signal_space.k_min,
                k_max=self.signal_space.k_max
            )

            tasks[ind] = InitialConditionsTask(
                t=self.t,
                d=self.d,
                signal_space=signal_space,
                depth=depth,
                grid_search_args=grid_search_args,
                nuts_args=nuts_args,
                perform_minimize=self.perform_minimize,
                path=path / f"subband_{ind + 1}r{subband_count}"
            )

        task_results_by_index = {}

        context = mp.get_context("spawn")
        manager = context.Manager()
        core_queue = manager.Queue()

        if hasattr(os, "sched_getaffinity"):
            available_core_ids = sorted(os.sched_getaffinity(0))
            available_cores = len(available_core_ids)
        else:
            available_core_ids = list(range(os.cpu_count() or 1))
            available_cores = len(available_core_ids)

        cores_per_worker = max(
            1,
            available_cores // self.max_workers,
        )

        required_cores = self.max_workers * cores_per_worker

        available_core_ids = psutil.Process().cpu_affinity()
        available_cores = len(available_core_ids)

        required_cores = self.max_workers * cores_per_worker

        if available_cores < required_cores:
            raise RuntimeError(
                f"Need {required_cores} cores, "
                f"but only {available_cores} are available."
            )

        for worker_index in range(self.max_workers):
            start = worker_index * cores_per_worker
            stop = start + cores_per_worker

            core_queue.put(available_core_ids[start:stop])

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(core_queue,),
        ) as executor:
            future_to_index = {
                executor.submit(
                    run_initial_conditions_worker_wrapper,
                    task,
                ): i
                for i, task in enumerate(tasks)
            }

            for future in as_completed(future_to_index):
                pbar.update(1)
                task_index = future_to_index[future]

                try:
                    result = future.result()

                    if result is not None:
                        result["subband_index"] = task_index
                        task_results_by_index[task_index] = result

                except BaseException as exc:
                    traceback_text = traceback.format_exc()

                    print(
                        f"\nWorker failed with {type(exc).__name__}: {exc}\n"
                        f"{traceback_text}",
                        flush=True,
                    )

                    raise RuntimeError(
                        f"{type(exc).__name__}: {exc}\n\n"
                        f"Worker traceback:\n{traceback_text}"
                    ) from None

        ordered_results = [
            task_results_by_index[i]
            for i in sorted(task_results_by_index)
        ]

        signals_init = list(
            chain.from_iterable(
                result["signals"]
                for result in ordered_results
            )
        )

        signals_bw_init = list(
            chain.from_iterable(
                result["signals_bw"]
                for result in ordered_results
            )
        )

        signals_and_bw = sorted(
            zip(signals_init, signals_bw_init),
            key=lambda pair: pair[0][0],  
        )

        self.signals_init, self.signals_bw_init = map(
            list,
            zip(*signals_and_bw)
        )

        self.signals_by_subband = {
            result["subband_index"]: {
                "f_min": result["f_min"],
                "f_max": result["f_max"],
                "signals": result["signals"],
                "signals_bw": result["signals_bw"],
                "noise_variance": result["noise_variance"],
                "snr": result["snr"],
                "glob_ll": result["glob_ll"],
                "reason": result["reason"]
            }
            for result in ordered_results
        }

        log_utils.save_initialize_csv(path / "all_subband_results.csv", signals_by_subband=self.signals_by_subband)

        create_deliverables(path, self.t, self.d, self.signals_init, self.signal_space)

        if self.signals_init:
            print(f"\nFound {len(self.signals_init)} signals!")
        else:
            print(f"\nNo signals found in data.")
            return
        
        return

                        
    def sample(
            self,
            signals: Any,
            signals_bw: Any,
            nuts_args: NUTSArgs,
            signals_per_block: int = 10,
            fill_order: int = 0,
            cores_per_worker: int = 1,
    ):
        print("Sampling...")

        path = self.path / "sample"
        path.mkdir(parents=True, exist_ok=True)
        n_signals = len(signals)
        block_size = signals_per_block
        stride = block_size // (2 ** fill_order)

        if stride <= 0:
            raise ValueError("stride must be at least 1")

        blocks = max(
            1,
            math.ceil((n_signals - block_size) / stride) + 1
        )

        tasks = []

        pbar = tqdm(total=blocks)

        for ind in range(blocks):
            start = int(ind * stride)
            stop = min(start + block_size, n_signals)

            signal_indices = list(range(start, stop))
            signal_block = signals[start:stop]
            signals_bw_block = signals_bw[start:stop]

            if ind == 0:
                f_min = self.signal_space.f_min
            else:
                # Boundary computed from the previous block's last signal
                # and this block's first signal
                previous_stop = min(start, n_signals)
                f_min = (
                    signals[previous_stop - 1][0] +
                    signals[previous_stop][0]
                ) / 2

            if ind == blocks - 1 or stop >= n_signals:
                f_max = self.signal_space.f_max
            else:
                f_max = (
                    signals[stop - 1][0] +
                    signals[stop][0]
                ) / 2

            signal_space = SignalSpace(
                f_min=f_min,
                f_max=f_max,
                k_min=self.signal_space.k_min,
                k_max=self.signal_space.k_max
            )

            tasks.append(
                SampleTask(
                    t=self.t,
                    d=self.d,
                    signal_space=signal_space,
                    signals=signal_block,
                    signals_bw=signals_bw_block,
                    signal_indices=signal_indices,
                    nuts_args=nuts_args,
                    perform_minimize=self.perform_minimize,
                    path=path / f"block_{ind + 1}r{blocks}"   
                )
            )

        results_by_signal = defaultdict(list)

        context = mp.get_context("spawn")
        manager = context.Manager()
        core_queue = manager.Queue()

        if hasattr(os, "sched_getaffinity"):
            available_core_ids = sorted(os.sched_getaffinity(0))
            available_cores = len(available_core_ids)
        else:
            available_core_ids = list(range(os.cpu_count() or 1))
            available_cores = len(available_core_ids)

        cores_per_worker = max(
            1,
            available_cores // self.max_workers,
        )

        required_cores = self.max_workers * cores_per_worker

        available_core_ids = psutil.Process().cpu_affinity()
        available_cores = len(available_core_ids)

        required_cores = self.max_workers * cores_per_worker

        if available_cores < required_cores:
            raise RuntimeError(
                f"Need {required_cores} cores, "
                f"but only {available_cores} are available."
            )

        for worker_index in range(self.max_workers):
            start = worker_index * cores_per_worker
            stop = start + cores_per_worker

            core_queue.put(available_core_ids[start:stop])

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(core_queue,),
        ) as executor:
            future_to_task_index = {
                executor.submit(
                    run_sample_worker_wrapper,
                    task,
                ): task_index
                for task_index, task in enumerate(tasks)
            }

            for future in as_completed(future_to_task_index):
                pbar.update(1)

                task_index = future_to_task_index[future]

                try:
                    indexed_results = future.result()

                    for signal_index, result in indexed_results:
                        results_by_signal[signal_index].append({
                            "task_index": task_index,
                            "result": result,
                        })

                except BaseException as exc:
                    traceback_text = traceback.format_exc()

                    print(
                        f"\nWorker failed for task {task_index}: "
                        f"{type(exc).__name__}: {exc}\n"
                        f"{traceback_text}",
                        flush=True,
                    )

                    raise RuntimeError(
                        f"Task {task_index} failed with "
                        f"{type(exc).__name__}: {exc}\n\n"
                        f"Worker traceback:\n{traceback_text}"
                    ) from None
        
        self.results = results_by_signal
        self.signals_sample = utils.unpack_signal_results(self.results)
        self.signals_bw_sample = self.signals_bw_init

        uncertainties = bats.get_uncertainties(self.t, self.d, self.signals_sample)
        log_utils.save_report_csv(path / "report.csv", self.results, uncertainties)

        create_deliverables(path, self.t, self.d, self.signals_sample, self.signal_space)


    def reconcile(
            self, 
            signals: Any,
            signals_bw: Any,
            nuts_args: Optional[NUTSArgs] = None,
        ):
        "Reconciling degenerate signals..."

        path = self.path / "reconcile"
        path.mkdir(parents=True, exist_ok=True)

        if self.perform_minimize is True:
            signals = bats.minimize(self.t, self.d, self.signal_space, signals)

        signals, signals_bw = bats.reconcile_noise_floor(self.t, self.d, self.signal_space, signals, signals_bw, nuts_args=nuts_args)

        create_deliverables(path, self.t, self.d, signals, self.signal_space)


    def execute(
            self,
            subband_count: int = 2,
            subband_scaling_factor: float = 1,
            depth: int = 10,
            grid_search_args: Optional[GridSearchArgs] = None,
            nuts_args_init: Optional[NUTSArgs] = None,
            perform_minimize: bool = False,
            cores_per_initial_conditions_worker: int = 1,

            signals_per_block: int = 1,
            fill_order: int = 1,
            nuts_args_sample: Optional[NUTSArgs] = None,
            cores_per_sample_worker: int = 1,

            nuts_args_reconcile: Optional[NUTSArgs] = None,
    ):        
        if not grid_search_args:
            grid_search_args = self.default_grid_search_args
        if not nuts_args_sample:
            nuts_args_sample = self.default_nuts_args_sample
        self.perform_minimize = perform_minimize
        
        self.initialize(
            subband_count=subband_count,
            subband_scaling_factor=subband_scaling_factor,
            depth=depth,
            grid_search_args=grid_search_args,
            nuts_args=nuts_args_init,
            cores_per_worker=cores_per_initial_conditions_worker,
        )
        self.sample(
            signals=self.signals_init,
            signals_bw=self.signals_bw_init,
            signals_per_block=signals_per_block,
            fill_order=fill_order,
            nuts_args=nuts_args_sample,
            cores_per_worker=cores_per_sample_worker
        )
        self.reconcile(
            signals=self.signals_sample,
            signals_bw=self.signals_bw_sample,
            nuts_args=nuts_args_reconcile,
        )

        print("Dracula complete!")


if __name__ == "__main__":
    t = np.linspace(0, 100, 2000)

    f1 = 4
    k1 = 1e-2
    f2 = 4.5
    k2 = 4e-3
    f3 = 4.05
    k3 = 7e-3

    e = np.random.normal(loc=0.0, scale=1, size=len(t))
    d = (np.sin(2 * np.pi * f1 * t) * np.exp(-k1 * t) + 
         np.sin(2 * np.pi * f2 * t) * np.exp(-k2 * t) +
         np.sin(2 * np.pi * f3 * t) * np.exp(-k3 * t) + 
         2 * e)

    model = Dracula(
        t, 
        d,
        f_min=3,
        f_max=5,
        k_min=1e-4,
        k_max=3e-2,
        max_workers=4,
    )

    grid_search_args = GridSearchArgs(
        f_points=100,
        k_points=100
    )

    nuts_kwargs = dict()
    mcmc_kwargs = dict(
        num_warmup=20,
        num_samples=20,
        num_chains=2,
        progress_bar=True
    )
    run_kwargs = dict()
    nuts_args = NUTSArgs(
        seed=None,
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        run_kwargs=run_kwargs,
    )

    model.execute(
        subband_count=1, 
        subband_scaling_factor=0.5,
        grid_search_args=grid_search_args,
        cores_per_initial_conditions_worker=1,
        depth=5,

        signals_per_block=5,
        fill_order=0,
        nuts_args_sample=nuts_args,
        cores_per_sample_worker=2,
        perform_minimize=False,
    )
