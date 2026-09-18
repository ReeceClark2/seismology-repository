from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from itertools import chain
from dataclasses import dataclass, asdict, field, fields
from typing import Any, Optional
from pathlib import Path
from datetime import datetime
import traceback
import math
from collections import defaultdict

import numpy as np

import jax.numpy as jnp
from jax.typing import ArrayLike

from tqdm import tqdm

import bats
import utils


@dataclass
class SignalSpace:
    f_min: float
    f_max: float
    k_min: float
    k_max: float

@dataclass
class GridSearchArgs:
    f_points: int
    k_points: int

@dataclass
class NUTSArgs:
    nuts_kwargs: dict[str, Any] = field(default_factory=dict)
    mcmc_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)

@dataclass
class InitialConditionsTask:
    t: ArrayLike
    d: ArrayLike
    signal_space: SignalSpace
    depth: int
    grid_search_args: GridSearchArgs
    nuts_args: NUTSArgs
    path: str

def run_initial_conditions_worker(t, d, signal_space, depth, grid_search_args, nuts_args, path):
    path.mkdir(parents=True, exist_ok=True)

    subband_t = t.copy()
    subband_d = utils.filter(subband_t, d.copy(), signal_space.f_min, signal_space.f_max)

    t_max = jnp.log(0.2) / (-signal_space.k_min)

    mask = t < t_max

    subband_t = subband_t[mask]
    subband_d = subband_d[mask]

    utils.plot_time_series(path / "raw_time_series.png", subband_t, subband_d, "Original Time Series")

    signals = []
    signal_bw = ((signal_space.f_max - signal_space.f_min) / 4, (jnp.log(signal_space.k_max) - jnp.log(signal_space.k_min)) / 4)
    noise_variances = []
    snrs = []
    
    signal_candidate, probability_surface = bats.grid_search(
        subband_t, 
        subband_d, 
        grid_search_args.f_points,
        signal_space.f_min, 
        signal_space.f_max,
        grid_search_args.k_points,
        signal_space.k_min, 
        signal_space.k_max,
        return_probability_surface=True
    )
    utils.plot_probability_surface(path / f"{len(signals) + 1}_probability_surface.png", probability_surface, f"Probability Surface of Signal {len(signals) + 1}")

    if nuts_args:
        rng_key_value = len(subband_t)
        signals_temp = tuple(x.copy() for x in signal_candidate)
        signal_candidate = bats.nuts(
            subband_t, 
            subband_d, 
            signal_candidate,
            signal_bw,
            nuts_args.nuts_kwargs,
            nuts_args.mcmc_kwargs,
            nuts_args.run_kwargs,
            rng_key_value
        )[0]

        utils.plot_signal_space(path / f"{len(signals) + 1}_signal_space.png", signal_candidate, f"Signal Space of {len(signals) + 1} Signals", signals_0=signals_temp, signals_bw=signal_bw, f_min=signal_space.f_min, f_max=signal_space.f_max, k_min=signal_space.k_min, k_max=signal_space.k_max)
    else:
        utils.plot_signal_space(path / f"{len(signals) + 1}_signal_space.png", signal_candidate, f"Signal Space of {len(signals) + 1} Signals", f_min=signal_space.f_min, f_max=signal_space.f_max, k_min=signal_space.k_min, k_max=signal_space.k_max)

    noise_variances.append(bats.get_noise_variance(subband_t, subband_d, signal_candidate))
    snrs.append(bats.get_snr(subband_t, subband_d, signal_candidate))

    signal_detected = utils.is_signal_detected(probability_surface)
    if not signal_detected:
        return

    signals.append(signal_candidate)
    glob_ll_0 = bats.get_glob_ll(t[mask], d[mask], signals)

    reason = "depth"
    while True:
        signal_detected = False
        model = bats.get_model(subband_t, subband_d, signals)
        residual = subband_d - model

        utils.plot_time_series(path / f"{len(signals)}_signal_time_series.png", subband_t, subband_d, f"Time Series for {len(signals)} Signal Model", model)

        signal_candidate, probability_surface = bats.grid_search(
            subband_t, 
            residual, 
            grid_search_args.f_points,
            signal_space.f_min, 
            signal_space.f_max,
            grid_search_args.k_points,
            signal_space.k_min, 
            signal_space.k_max,
            return_probability_surface=True
        )
        utils.plot_probability_surface(path / f"{len(signals) + 1}_probability_surface.png", probability_surface, f"Probability Surface of Signal {len(signals) + 1}")

        signals_with_candidate = list(signals) + [signal_candidate]

        if nuts_args:
            signals_temp = signals_with_candidate.copy()
            signals_with_candidate = bats.nuts(
                subband_t, 
                subband_d, 
                signals_with_candidate,
                signal_bw,
                nuts_args.nuts_kwargs,
                nuts_args.mcmc_kwargs,
                nuts_args.run_kwargs,
                rng_key_value
            )
            signal_candidate = bats.nuts(
                subband_t, 
                residual, 
                signal_candidate,
                signal_bw,
                nuts_args.nuts_kwargs,
                nuts_args.mcmc_kwargs,
                nuts_args.run_kwargs,
                rng_key_value
            )
            
            utils.plot_signal_space(path / f"{len(signals) + 1}_signal_space.png", signals_with_candidate, f"Signal Space of {len(signals) + 1} Signals", signals_0=signals_temp, signals_bw=signal_bw, f_min=signal_space.f_min, f_max=signal_space.f_max, k_min=signal_space.k_min, k_max=signal_space.k_max)
        else:
            utils.plot_signal_space(path / f"{len(signals) + 1}_signal_space.png", signals_with_candidate, f"Signal Space of {len(signals) + 1} Signals", f_min=signal_space.f_min, f_max=signal_space.f_max, k_min=signal_space.k_min, k_max=signal_space.k_max)

        glob_ll_1 = bats.get_glob_ll(t[mask], d[mask], signals_with_candidate)
        delta_glob_ll = glob_ll_1 - glob_ll_0

        if delta_glob_ll < 0:
            reason = "glob_ll"
            break

        log_prob = bats.get_log_prob(subband_t, residual, signal_candidate)
        signal_detected = utils.is_signal_detected(probability_surface, log_prob)
        if not signal_detected:
            reason = "rcr"
            break

        glob_ll_0 = glob_ll_1

        signals = signals_with_candidate
        noise_variances.append(bats.get_noise_variance(subband_t, subband_d, signals))
        snrs.append(bats.get_snr(subband_t, subband_d, signals))

        if len(signals) >= depth:
            break

    utils.save_subband_csv(path / "subband_results.csv", signals, noise_variances, snrs)

    signals_bw = [tuple(signal_bw) for _ in signals]

    noise_variance = bats.get_noise_variance(t, d, signals)
    snr = bats.get_snr(t, d, signals)

    return {
        "f_min": signal_space.f_min,
        "f_max": signal_space.f_max,
        "signals": signals,
        "signals_bw": signals_bw,
        "noise_variance": noise_variance,
        "snr": snr,
        "reason": reason,
    }

def run_initial_conditions_worker_wrapper(
    config: InitialConditionsTask,
):
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
    t: ArrayLike
    d: ArrayLike
    signal_space: SignalSpace
    signals: Any
    signals_bw: Any
    signal_indices: list[int]
    nuts_args: NUTSArgs
    path: str

def run_sample_worker(
    t,
    d,
    signal_space,
    signals,
    signals_bw,
    signal_indices,
    nuts_args,
    path
):
    path.mkdir(parents=True, exist_ok=True)

    d = utils.filter(t, d, signal_space.f_min, signal_space.f_max)

    t_max = jnp.log(0.2) / (-signal_space.k_min)

    mask = t < t_max

    t = t[mask]
    d = d[mask]

    utils.plot_time_series(path / "raw_time_series.png", t, d, "Original Time Series")

    rng_key_value = len(t)
    signals_0 = signals.copy()
    signals = bats.nuts(
        t,
        d,
        signals,
        signals_bw,
        nuts_args.nuts_kwargs,
        nuts_args.mcmc_kwargs,
        nuts_args.run_kwargs,
        rng_key_value
    )

    utils.plot_signal_space(path / f"{len(signals) + 1}_signal_space.png", signals, f"Signal Space of {len(signals) + 1} Signals", signals_0=signals_0, signals_bw=signals_bw, f_min=signal_space.f_min, f_max=signal_space.f_max, k_min=signal_space.k_min, k_max=signal_space.k_max)

    model = bats.get_model(t, d, signals)
    utils.plot_time_series(path / f"model_time_series.png", t, d, "Model Time Series", model)
    utils.save_block_csv(path / "block_results.csv", signals)

    if len(signals) != len(signal_indices):
        raise ValueError(
            "Number of NUTS results does not match number of input signals: "
            f"{len(signals)=}, {len(signal_indices)=}"
        )

    return list(zip(signal_indices, signals))

def run_sample_worker_wrapper(config: SampleTask):
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

class Dracula():
    def __init__(
            self, 
            t: ArrayLike, 
            d: ArrayLike,
            f_min: float,
            f_max: float,
            k_min: float,
            k_max: float,
            max_workers: int=1
        ):
        self.t = jnp.asarray(t)
        self.d = jnp.asarray(d)

        self.signal_space = SignalSpace(
            f_min=f_min,
            f_max=f_max,
            k_min=k_min,
            k_max=k_max
        )

        self.max_workers = max_workers

        path = Path.cwd() / "dracula"

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = path.parent / f"{path.name}_{timestamp}"
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
            run_kwargs=run_kwargs
        )

    def initialize(
            self,
            subband_count: int,
            subband_scaling_factor: float,
            depth: int,
            grid_search_args: GridSearchArgs,
            nuts_args: NUTSArgs,
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
                path=path / f"subband_{ind + 1}r{subband_count}"
            )

        task_results_by_index = {}

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context("spawn"),
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
                "reason": result["reason"]
            }
            for result in ordered_results
        }

        utils.save_initialize_csv(path / "initialize_results.csv", signals_by_subband=self.signals_by_subband)

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
            signals_per_block: int,
            fill_order: int,
            nuts_args: NUTSArgs
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
                    path=path / f"block_{ind}r{blocks}"
                )
            )

        results_by_signal = defaultdict(list)

        with ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=mp.get_context("spawn"),
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

    def report(
            self,
            results
    ):
        "Creating report..."

        path = self.path / "report"
        path.mkdir(parents=True, exist_ok=True)

        utils.save_report_csv(path / "report.csv", results)

        signals = utils.unpack_signal_results(results)

        noise_variance = bats.get_noise_variance(self.t, self.d, signals)
        snr = bats.get_snr(self.t, self.d, signals)
        
        utils.save_report_txt(path / "report.txt", len(signals), noise_variance, snr)

        model = bats.get_model(self.t, self.d, signals)
        utils.plot_time_series(path / "model_time_series.png", self.t, self.d, "Model Time Series", model)
        utils.plot_signal_space(path / "signal_space.png", signals, "Signal Space", self.signals_init, self.signals_bw_init[0], f_min=self.signal_space.f_min, f_max=self.signal_space.f_max, k_min=self.signal_space.k_min, k_max=self.signal_space.k_max)        
        
    def execute(
            self,
            subband_count: int = 1,
            subband_scaling_factor: float = 1,
            depth: int = 10,
            grid_search_args:  Optional[GridSearchArgs] = None,
            nuts_args_init: Optional[NUTSArgs] = None,

            signals_per_block: int = 1,
            fill_order: int = 0,
            nuts_args_sample:  Optional[NUTSArgs] = None,
    ):        
        if not grid_search_args:
            grid_search_args = self.default_grid_search_args
        if not nuts_args_sample:
            nuts_args_sample = self.default_nuts_args_sample
        
        self.initialize(
            subband_count=subband_count,
            subband_scaling_factor=subband_scaling_factor,
            depth=depth,
            grid_search_args=grid_search_args,
            nuts_args=nuts_args_init,
        )
        self.sample(
            signals=self.signals_init,
            signals_bw=self.signals_bw_init,
            signals_per_block=signals_per_block,
            fill_order=fill_order,
            nuts_args=nuts_args_sample
        )
        self.report(
            self.results
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

    e = np.random.normal(loc=0.0, scale=3, size=len(t))
    d = (np.sin(2 * np.pi * f1 * t) * np.exp(-k1 * t) + 
         np.sin(2 * np.pi * f2 * t) * np.exp(-k2 * t) +
         np.sin(2 * np.pi * f3 * t) * np.exp(-k3 * t) + 
         e)


    model = Dracula(
        t, 
        d,
        f_min=3,
        f_max=5,
        k_min=1e-4,
        k_max=3e-2,
        max_workers=4
    )

    grid_search_args = GridSearchArgs(
        f_points=100,
        k_points=100
    )

    nuts_kwargs = dict()
    mcmc_kwargs = dict(
        num_warmup=20,
        num_samples=20,
        num_chains=1,
        progress_bar=True
    )
    run_kwargs = dict()
    nuts_args = NUTSArgs(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        run_kwargs=run_kwargs,
    )

    model.execute(
        subband_count=1, 
        subband_scaling_factor=0.5,
        grid_search_args=grid_search_args,
        nuts_args_init=nuts_args,

        signals_per_block=1,
        fill_order=0,
        nuts_args_sample=nuts_args
    )
