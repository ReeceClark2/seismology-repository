from __future__ import annotations

import math
import multiprocessing
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, sosfilt

from bats import (
    BATS,
    BATSResult,
    as_1d_float,
    broadcast_bandwidth,
    progress_bar_label,
    split_numpyro_kwargs,
)

_PROGRESS_MAX: int = 1


def init_parallel_worker(tqdm_lock: Any, max_cores: int) -> None:
    """Initializer for process-pool workers: local tqdm lock, no Manager proxies."""
    global _PROGRESS_MAX
    _PROGRESS_MAX = max(1, int(max_cores))
    try:
        import tqdm
        from tqdm.auto import tqdm as tqdm_auto

        # The monitor thread plus a Manager.RLock causes BrokenPipeError on exit.
        tqdm.tqdm.monitor_interval = 0
        tqdm_auto.monitor_interval = 0
        if tqdm_lock is not None:
            tqdm.tqdm.set_lock(tqdm_lock)
            tqdm_auto.set_lock(tqdm_lock)
    except Exception:
        pass


def acquire_progress_slot() -> int:
    ident = multiprocessing.current_process()._identity
    if ident:
        return int(ident[0] - 1) % _PROGRESS_MAX
    return 0


def release_progress_slot(position: int) -> None:
    return


@dataclass
class BATSTask:
    """Picklable payload for one process-pool BATS worker.

    Sampler knobs live in nested dicts so new NumPyro parameters can be
    added without changing this dataclass's required fields.
    """
    t: ArrayLike
    d: ArrayLike
    f_init: ArrayLike
    k_init: ArrayLike
    f_bw: ArrayLike
    k_bw: ArrayLike
    W: int
    S: int
    seed: int
    nuts_kwargs: dict[str, Any] = field(default_factory=dict)
    mcmc_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)
    n_signals: int = 0
    freq_lo: int = 1
    freq_hi: int = 1
    prior_n_std: float = 5.0


@dataclass
class ColonyJob:
    """Picklable payload for building one N-signal Colony's BATS tasks."""
    t: ArrayLike
    d: ArrayLike
    f_init: ArrayLike
    k_init: ArrayLike
    f_per_worker: int
    f_bw: float | ArrayLike | None
    k_bw: float | ArrayLike | None
    W: int
    S: int
    n_signals: int
    prior_n_std: float = 5.0
    nuts_kwargs: dict[str, Any] = field(default_factory=dict)
    mcmc_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)


def run_colony_worker(job: ColonyJob) -> tuple[int, list[BATSTask]]:
    colony = Colony(job.t, job.d, job.f_init, job.k_init)
    tasks = colony.get_tasks(
        job.f_per_worker,
        job.f_bw,
        job.k_bw,
        job.W,
        job.S,
        prior_n_std=job.prior_n_std,
        nuts_kwargs=job.nuts_kwargs,
        mcmc_kwargs=job.mcmc_kwargs,
        run_kwargs=job.run_kwargs,
    )
    for task in tasks:
        task.n_signals = job.n_signals
    return job.n_signals, tasks


def run_bats_worker(task: BATSTask) -> BATSResult:
    slot = acquire_progress_slot()
    try:
        n_signals = task.n_signals or int(len(task.f_init))
        desc = progress_bar_label(task.freq_lo, task.freq_hi, n_signals)
        model = BATS(task.t, task.d, task.f_init, task.k_init)
        return model.run_NUTS(
            task.f_bw,
            task.k_bw,
            task.W,
            task.S,
            task.seed,
            progress_desc=desc,
            progress_position=slot + 1,
            prior_n_std=task.prior_n_std,
            nuts_kwargs=task.nuts_kwargs,
            mcmc_kwargs=task.mcmc_kwargs,
            run_kwargs=task.run_kwargs,
        )
    finally:
        release_progress_slot(slot)


def infer_sampling_rate(t: ArrayLike) -> float:
    """Sampling frequency in Hz from a time vector given in seconds."""
    t_arr = np.asarray(t, dtype=float).ravel()
    if t_arr.size < 2:
        raise ValueError("t must contain at least two samples to infer sampling rate")
    dt = np.diff(t_arr)
    if np.any(dt <= 0):
        raise ValueError("t must be strictly increasing (seconds)")
    return float(1.0 / np.median(dt))


def butter_bandpass(
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Use second-order sections for numerical stability
    sos = butter(order, [low, high], btype="band", output="sos")
    return sos


def butter_bandpass_filter(
    data: ArrayLike,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, np.asarray(data))
    return y


class Colony:
    def __init__(
        self,
        t: ArrayLike,
        d: ArrayLike,
        f_init: ArrayLike,
        k_init: ArrayLike,
    ) -> None:
        self.t = as_1d_float(t, "t")
        self.d = as_1d_float(d, "d")
        if self.t.shape[0] != self.d.shape[0]:
            raise ValueError(
                f"t and d must have the same length, got {self.t.shape[0]} and {self.d.shape[0]}"
            )

        f_init = as_1d_float(f_init, "f_init")
        k_init = as_1d_float(k_init, "k_init")
        if f_init.shape[0] != k_init.shape[0]:
            raise ValueError(
                "f_init and k_init must have the same length, "
                f"got {f_init.shape[0]} and {k_init.shape[0]}"
            )

        self._order = jnp.argsort(f_init)
        self.f_init = f_init[self._order]
        self.k_init = k_init[self._order]

    def get_tasks(
        self,
        f_per_worker: int,
        f_bw: float | ArrayLike | None = None,
        k_bw: float | ArrayLike | None = None,
        W: int = 1_000,
        S: int = 2_000,
        prior_n_std: float = 5.0,
        **kwargs: Any,
    ) -> list[BATSTask]:
        n = int(self.f_init.shape[0])
        if f_per_worker < 1:
            raise ValueError(f"f_per_worker must be >= 1, got {f_per_worker}")

        if f_bw is None:
            f_bw = 1e-3
        if k_bw is None:
            k_bw = 1e-5
        if prior_n_std <= 0:
            raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")

        f_bw = broadcast_bandwidth(f_bw, n, "f_bw")[self._order]
        k_bw = broadcast_bandwidth(k_bw, n, "k_bw")[self._order]

        nuts_kwargs, mcmc_kwargs, run_kwargs = split_numpyro_kwargs(kwargs)

        workers = math.ceil(n / f_per_worker) if n else 0
        chunk_size = int(f_per_worker)
        tasks: list[BATSTask] = []
        fs_sample = infer_sampling_rate(self.t)
        nyq = 0.5 * fs_sample

        for i in range(workers):
            start = i * chunk_size
            end = min(start + chunk_size, n)

            f_chunk = self.f_init[start:end]
            k_chunk = self.k_init[start:end]
            f_bw_chunk = f_bw[start:end]
            k_bw_chunk = k_bw[start:end]

            if len(f_chunk) == 0:
                continue

            lowcut = float(self.f_init[start] - (5 * f_bw[start]))
            highcut = float(self.f_init[end - 1] + (5 * f_bw[end - 1]))
            lowcut = min(max(lowcut, 1e-12), nyq * 0.999)
            highcut = min(highcut, nyq * 0.999)
            if lowcut < highcut:
                filtered_d = butter_bandpass_filter(
                    self.d,
                    lowcut,
                    highcut,
                    fs_sample,
                    order=6,
                )
            else:
                filtered_d = np.asarray(self.d)

            tasks.append(
                BATSTask(
                    t=self.t,
                    d=filtered_d,
                    f_init=f_chunk,
                    k_init=k_chunk,
                    f_bw=f_bw_chunk,
                    k_bw=k_bw_chunk,
                    W=W,
                    S=S,
                    seed=i,
                    nuts_kwargs=nuts_kwargs,
                    mcmc_kwargs=mcmc_kwargs,
                    run_kwargs=run_kwargs,
                    n_signals=n,
                    freq_lo=start + 1,
                    freq_hi=end,
                    prior_n_std=prior_n_std,
                )
            )

        return tasks
