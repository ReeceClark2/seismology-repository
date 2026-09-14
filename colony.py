from __future__ import annotations

import math
import multiprocessing
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

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
    resolve_imposed_surface,
    split_numpyro_kwargs,
)

_PROGRESS_MAX: int = 1

ProgressMode = Literal["none", "main", "detailed"]
PROGRESS_MODES: tuple[str, ...] = ("none", "main", "detailed")


def resolve_progress_mode(progress_mode: str) -> ProgressMode:
    mode = str(progress_mode)
    if mode not in PROGRESS_MODES:
        raise ValueError(
            "progress_mode must be one of "
            f"{PROGRESS_MODES}, got {progress_mode!r}"
        )
    return mode  # type: ignore[return-value]


def nested_progress_enabled(progress_mode: str) -> bool:
    return resolve_progress_mode(progress_mode) == "detailed"


def resolve_signals_per_worker(
    signals_per_worker: int | None = None,
    f_per_worker: int | None = None,
) -> int:
    """Return ``signals_per_worker``, accepting deprecated ``f_per_worker``."""
    if signals_per_worker is not None and f_per_worker is not None:
        raise ValueError(
            "Provide only one of signals_per_worker or f_per_worker"
        )
    if signals_per_worker is None and f_per_worker is None:
        raise TypeError("signals_per_worker is required")
    if f_per_worker is not None:
        warnings.warn(
            "f_per_worker is deprecated; use signals_per_worker",
            DeprecationWarning,
            stacklevel=3,
        )
        signals_per_worker = f_per_worker
    value = int(signals_per_worker)
    if value < 1:
        raise ValueError(f"signals_per_worker must be >= 1, got {value}")
    return value


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
    imposed_surface: str = "gaussian"
    progress_mode: str = "detailed"
    n_chains: int = 1


@dataclass
class ColonyJob:
    """Picklable payload for building one N-signal Colony's BATS tasks."""
    t: ArrayLike
    d: ArrayLike
    f_init: ArrayLike
    k_init: ArrayLike
    signals_per_worker: int
    f_bw: float | ArrayLike | None
    k_bw: float | ArrayLike | None
    W: int
    S: int
    n_signals: int
    prior_n_std: float = 5.0
    imposed_surface: str = "gaussian"
    progress_mode: str = "detailed"
    n_chains: int = 1
    nuts_kwargs: dict[str, Any] = field(default_factory=dict)
    mcmc_kwargs: dict[str, Any] = field(default_factory=dict)
    run_kwargs: dict[str, Any] = field(default_factory=dict)


def run_colony_worker(job: ColonyJob) -> tuple[int, list[BATSTask]]:
    colony = Colony(job.t, job.d, job.f_init, job.k_init)
    tasks = colony.get_tasks(
        job.signals_per_worker,
        job.f_bw,
        job.k_bw,
        job.W,
        job.S,
        prior_n_std=job.prior_n_std,
        imposed_surface=job.imposed_surface,
        progress_mode=job.progress_mode,
        n_chains=job.n_chains,
        nuts_kwargs=job.nuts_kwargs,
        mcmc_kwargs=job.mcmc_kwargs,
        run_kwargs=job.run_kwargs,
    )
    for task in tasks:
        task.n_signals = job.n_signals
    return job.n_signals, tasks


def _minimum_finite_potential_energy(result: BATSResult) -> float:
    pe = np.asarray(result.extras.get("potential_energy"), dtype=float).ravel()
    finite = pe[np.isfinite(pe)]
    if finite.size == 0:
        return float("inf")
    return float(np.min(finite))


def run_bats_worker(task: BATSTask) -> BATSResult:
    slot = acquire_progress_slot()
    try:
        n_signals = task.n_signals or int(len(task.f_init))
        show_nested = nested_progress_enabled(task.progress_mode)
        desc = progress_bar_label(task.freq_lo, task.freq_hi, n_signals)
        mcmc_kwargs = dict(task.mcmc_kwargs)
        if not show_nested:
            mcmc_kwargs["progress_bar"] = False
        mcmc_kwargs["num_chains"] = 1
        n_chains = max(1, int(getattr(task, "n_chains", 1)))
        model = BATS(task.t, task.d, task.f_init, task.k_init)
        best: BATSResult | None = None
        best_pe = float("inf")
        diagnostics: list[dict[str, Any]] = []
        for chain in range(n_chains):
            seed = int(task.seed) * 1009 + chain
            chain_desc = desc if n_chains == 1 else f"{desc} chain {chain + 1}/{n_chains}"
            result = model.run_nuts(
                task.f_bw,
                task.k_bw,
                task.W,
                task.S,
                seed,
                progress_desc=chain_desc if show_nested else None,
                progress_position=(slot + 1) if show_nested else None,
                prior_n_std=task.prior_n_std,
                imposed_surface=task.imposed_surface,
                nuts_kwargs=task.nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                run_kwargs=task.run_kwargs,
            )
            min_pe = _minimum_finite_potential_energy(result)
            diagnostics.append(
                {
                    "chunk_seed": int(task.seed),
                    "chain": int(chain),
                    "seed": seed,
                    "freq_lo": int(task.freq_lo),
                    "freq_hi": int(task.freq_hi),
                    "min_potential_energy": min_pe
                    if np.isfinite(min_pe)
                    else None,
                    "n_finite_samples": int(
                        result.extras.get("n_finite_samples", 0)
                    ),
                }
            )
            if min_pe < best_pe:
                best_pe = min_pe
                best = result
        if best is None:
            raise RuntimeError("NUTS produced no chain results")
        best.extras["chain_diagnostics"] = diagnostics
        best.extras["n_chains"] = n_chains
        best.extras["selected_min_potential_energy"] = (
            best_pe if np.isfinite(best_pe) else None
        )
        return best
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
        signals_per_worker: int | None = None,
        f_bw: float | ArrayLike | None = None,
        k_bw: float | ArrayLike | None = None,
        W: int = 1_000,
        S: int = 2_000,
        prior_n_std: float = 5.0,
        unbounded: bool | None = None,
        imposed_surface: str | None = None,
        f_per_worker: int | None = None,
        progress_mode: str = "detailed",
        n_chains: int = 1,
        **kwargs: Any,
    ) -> list[BATSTask]:
        n = int(self.f_init.shape[0])
        chunk_size = resolve_signals_per_worker(
            signals_per_worker=signals_per_worker,
            f_per_worker=f_per_worker,
        )
        progress_mode = resolve_progress_mode(progress_mode)
        surface = resolve_imposed_surface(
            imposed_surface=imposed_surface,
            unbounded=unbounded,
        )

        if f_bw is None:
            f_bw = 1e-3
        if k_bw is None:
            k_bw = 1e-5
        if prior_n_std <= 0:
            raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")

        f_bw = broadcast_bandwidth(f_bw, n, "f_bw")[self._order]
        k_bw = broadcast_bandwidth(k_bw, n, "k_bw")[self._order]

        nuts_kwargs, mcmc_kwargs, run_kwargs = split_numpyro_kwargs(kwargs)
        n_chains = max(1, int(n_chains))
        mcmc_kwargs = dict(mcmc_kwargs)
        mcmc_kwargs["num_chains"] = 1

        workers = math.ceil(n / chunk_size) if n else 0
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
                    imposed_surface=surface,
                    progress_mode=progress_mode,
                    n_chains=n_chains,
                )
            )

        return tasks
