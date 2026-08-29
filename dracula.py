from __future__ import annotations

import concurrent.futures
import csv
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from bats import (
    BATSResult,
    StatisticsResult,
    as_1d_float,
    broadcast_bandwidth,
    get_model,
    get_statistics,
    prefix_bandwidth,
    rank_by_power,
    split_numpyro_kwargs,
)
from colony import (
    ColonyJob,
    init_parallel_worker,
    run_bats_worker,
    run_colony_worker,
)

multiprocessing.set_start_method("spawn", force=True)


def _np(value: Any) -> np.ndarray:
    return np.asarray(value)


def _nearest_power(p_spec: ArrayLike, frequencies: ArrayLike) -> np.ndarray:
    f_grid, p_grid = _np(p_spec).T
    frequencies = _np(frequencies).ravel()
    idx = np.array([np.argmin(np.abs(f_grid - f)) for f in frequencies])
    return p_grid[idx]


def _resolve_output_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    path = Path(output_dir) if output_dir is not None else Path.cwd() / "dracula_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_signal_csv(path: Path, stats: StatisticsResult) -> None:
    order = np.argsort(_np(stats.fs))
    rows = zip(
        _np(stats.fs)[order],
        _np(stats.f_unc)[order],
        _np(stats.ks)[order],
        _np(stats.k_unc)[order],
    )
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequencies",
                "frequency_uncertainties",
                "decay_rates",
                "decay_rate_uncertainties",
            ]
        )
        for row in rows:
            writer.writerow([float(v) for v in row])


def _write_global_csv(path: Path, by_n: dict[int, StatisticsResult | BATSResult | None]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["N", "SNR", "variance", "glob_LL"])
        for n in sorted(by_n):
            result = by_n[n]
            if not isinstance(result, StatisticsResult):
                continue
            writer.writerow(
                [
                    int(n),
                    float(result.SNR),
                    float(result.variance),
                    float(result.glob_LL),
                ]
            )


def _plot_timeseries(
    path: Path,
    t: ArrayLike,
    d: ArrayLike,
    fs: ArrayLike,
    ks: ArrayLike,
    SNR: ArrayLike,
    variance: ArrayLike,
    n: int,
) -> None:
    t_np = _np(t)
    d_np = _np(d)
    model = _np(get_model(t, d, fs, ks))
    residual = d_np - model

    fig, axes = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12, 7),
        gridspec_kw={"height_ratios": [2, 1]},
    )
    axes[0].plot(t_np, d_np, color="black", lw=0.7, label="Data")
    axes[0].plot(t_np, model, color="C0", lw=0.9, alpha=0.85, label="Model (h·H)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"N = {n} signal model")
    axes[0].text(
        0.02,
        0.95,
        f"SNR = {float(SNR):.4g}\nvariance = {float(variance):.4g}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[1].plot(t_np, residual, color="C3", lw=0.7, label="Residual (data − model)")
    axes[1].axhline(0.0, color="gray", lw=0.6, alpha=0.7)
    axes[1].set_ylabel("Residual")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_power_spectrum(
    path: Path,
    before: StatisticsResult,
    after: StatisticsResult,
    f_init: ArrayLike,
    n: int,
) -> None:
    f_before, p_before = _np(before.p_spec).T
    f_after, p_after = _np(after.p_spec).T
    order_b = np.argsort(f_before)
    order_a = np.argsort(f_after)

    f_init_np = _np(f_init).ravel()
    f_found = _np(after.fs).ravel()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        f_before[order_b],
        p_before[order_b],
        color="0.45",
        lw=1.0,
        label="Power spectrum (before sampling)",
    )
    ax.plot(
        f_after[order_a],
        p_after[order_a],
        color="C0",
        lw=1.1,
        label="Power spectrum (after sampling)",
    )
    ax.scatter(
        f_init_np,
        _nearest_power(before.p_spec, f_init_np),
        s=36,
        color="black",
        zorder=3,
        label="Original frequencies",
    )
    ax.scatter(
        f_found,
        _nearest_power(after.p_spec, f_found),
        s=42,
        marker="x",
        color="C3",
        zorder=4,
        label="Found frequencies",
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"N = {n} power spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_outputs(
    output_dir: Path,
    t: ArrayLike,
    d: ArrayLike,
    by_n: dict[int, StatisticsResult | BATSResult | None],
    f_init_by_n: dict[int, Any],
    k_init_by_n: dict[int, Any],
) -> None:
    _write_global_csv(output_dir / "global_stats.csv", by_n)

    for n in sorted(by_n):
        result = by_n[n]
        if not isinstance(result, StatisticsResult):
            continue

        stem = f"N{n:03d}"
        _write_signal_csv(output_dir / f"{stem}_signals.csv", result)
        _plot_timeseries(
            output_dir / f"{stem}_timeseries.png",
            t,
            d,
            result.fs,
            result.ks,
            result.SNR,
            result.variance,
            n,
        )
        before = get_statistics(t, d, f_init_by_n[n], k_init_by_n[n])
        _plot_power_spectrum(
            output_dir / f"{stem}_power_spectrum.png",
            before,
            result,
            f_init_by_n[n],
            n,
        )


@dataclass
class DraculaResult:
    """Dispatch outputs keyed by signal count ``N``.

    Use ``result[n]`` for the N-signal model, or ``as_list()`` for
    ``min_signals`` … ``max_signals`` order. ``extras`` holds anything
    added later without changing required fields.
    """
    by_n: dict[int, StatisticsResult | BATSResult | None]
    min_signals: int
    max_signals: int
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, n: int) -> StatisticsResult | BATSResult | None:
        return self.by_n[n]

    def __iter__(self):
        for n in range(self.min_signals, self.max_signals + 1):
            yield self.by_n[n]

    def as_list(self) -> list[StatisticsResult | BATSResult | None]:
        return [self.by_n[s] for s in range(self.min_signals, self.max_signals + 1)]


class Dracula:
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

        self.f_init = as_1d_float(f_init, "f_init")
        self.k_init = as_1d_float(k_init, "k_init")
        if self.f_init.shape[0] != self.k_init.shape[0]:
            raise ValueError(
                "f_init and k_init must have the same length, "
                f"got {self.f_init.shape[0]} and {self.k_init.shape[0]}"
            )

    def _prepare_signals(
        self,
        f_bw: float | ArrayLike | None,
        k_bw: float | ArrayLike | None,
        sort_signals: bool,
    ) -> tuple[Any, Any, float | Any | None, float | Any | None, dict[str, Any]]:
        f_work = self.f_init
        k_work = self.k_init
        f_bw_work = f_bw
        k_bw_work = k_bw
        extras: dict[str, Any] = {"sort_signals": sort_signals}

        if sort_signals:
            order, powers, _ = rank_by_power(self.t, self.d, f_work, k_work)
            f_work = f_work[order]
            k_work = k_work[order]
            n = int(f_work.shape[0])
            if f_bw is not None:
                f_bw_work = broadcast_bandwidth(f_bw, n, "f_bw")[order]
            if k_bw is not None:
                k_bw_work = broadcast_bandwidth(k_bw, n, "k_bw")[order]
            extras["sort_order"] = order
            extras["powers"] = powers[order]

        return f_work, k_work, f_bw_work, k_bw_work, extras

    def dispatch(
        self,
        f_per_worker: int,
        min_signals: int,
        max_signals: int,
        f_bw: float | ArrayLike | None = None,
        k_bw: float | ArrayLike | None = None,
        W: int = 1_000,
        S: int = 2_000,
        calc_stats: bool = True,
        max_cores: int | None = None,
        sort_signals: bool = True,
        output_dir: str | os.PathLike[str] | None = None,
        prior_n_std: float = 5.0,
        **kwargs: Any,
    ) -> DraculaResult:
        n_total = int(self.f_init.shape[0])
        if min_signals < 1:
            raise ValueError(f"min_signals must be >= 1, got {min_signals}")
        if max_signals < min_signals:
            raise ValueError(
                f"max_signals ({max_signals}) must be >= min_signals ({min_signals})"
            )
        if max_signals > n_total:
            raise ValueError(
                f"max_signals ({max_signals}) exceeds available signals ({n_total})"
            )
        if f_per_worker < 1:
            raise ValueError(f"f_per_worker must be >= 1, got {f_per_worker}")
        if prior_n_std <= 0:
            raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")

        f_work, k_work, f_bw_work, k_bw_work, sort_extras = self._prepare_signals(
            f_bw, k_bw, sort_signals
        )

        if max_cores is None:
            max_cores = max(1, (os.cpu_count() or 4) - 2)

        print(f"Limiting execution to {max_cores} concurrent workers.")

        nuts_kwargs, mcmc_kwargs, run_kwargs = split_numpyro_kwargs(kwargs)

        grouped_results: dict[int, list[BATSResult]] = {
            s: [] for s in range(min_signals, max_signals + 1)
        }
        tasks_remaining: dict[int, int | None] = {
            s: None for s in range(min_signals, max_signals + 1)
        }
        final_results: dict[int, StatisticsResult | BATSResult | None] = {
            s: None for s in range(min_signals, max_signals + 1)
        }
        f_init_by_n = {s: f_work[:s] for s in range(min_signals, max_signals + 1)}
        k_init_by_n = {s: k_work[:s] for s in range(min_signals, max_signals + 1)}

        n_models = max_signals - min_signals + 1
        colony_queue = list(range(min_signals, max_signals + 1))

        future_metadata: dict[Any, tuple[str, int]] = {}
        pending_futures: set[concurrent.futures.Future[Any]] = set()

        def submit_colony(executor: concurrent.futures.ProcessPoolExecutor) -> None:
            if not colony_queue:
                return
            signals = colony_queue.pop(0)
            job = ColonyJob(
                t=self.t,
                d=self.d,
                f_init=f_work[:signals],
                k_init=k_work[:signals],
                f_per_worker=f_per_worker,
                f_bw=prefix_bandwidth(f_bw_work, signals, "f_bw"),
                k_bw=prefix_bandwidth(k_bw_work, signals, "k_bw"),
                W=W,
                S=S,
                n_signals=signals,
                prior_n_std=prior_n_std,
                nuts_kwargs=nuts_kwargs,
                mcmc_kwargs=mcmc_kwargs,
                run_kwargs=run_kwargs,
            )
            future = executor.submit(run_colony_worker, job)
            future_metadata[future] = ("colony", signals)
            pending_futures.add(future)

        def launch_stats_if_ready(
            executor: concurrent.futures.ProcessPoolExecutor,
            signals: int,
        ) -> None:
            if tasks_remaining[signals] != 0 or not grouped_results[signals]:
                return
            colony_res = sorted(
                grouped_results[signals],
                key=lambda r: r.seed if r.seed is not None else 0,
            )
            flat_fs = jnp.concatenate([r.fs for r in colony_res])
            flat_ks = jnp.concatenate([r.ks for r in colony_res])
            combined = BATSResult(fs=flat_fs, ks=flat_ks)
            if calc_stats:
                stats_future = executor.submit(
                    get_statistics,
                    self.t,
                    self.d,
                    flat_fs,
                    flat_ks,
                )
                future_metadata[stats_future] = ("stats", signals)
                pending_futures.add(stats_future)
            else:
                final_results[signals] = combined

        tqdm.monitor_interval = 0
        ctx = multiprocessing.get_context("spawn")
        tqdm_lock = ctx.RLock()
        try:
            tqdm.set_lock(tqdm_lock)
        except Exception:
            pass

        pipeline = tqdm(
            total=n_models,
            desc="Dracula",
            position=0,
            leave=True,
            dynamic_ncols=True,
            unit="job",
        )

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_cores,
                initializer=init_parallel_worker,
                initargs=(tqdm_lock, max_cores),
            ) as executor:

                for _ in range(min(max_cores, n_models)):
                    submit_colony(executor)

                while pending_futures:
                    done, pending_futures = concurrent.futures.wait(
                        pending_futures,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )

                    for future in done:
                        task_type, signals = future_metadata.pop(future)

                        try:
                            if task_type == "colony":
                                _, colony_tasks = future.result()
                                tasks_remaining[signals] = len(colony_tasks)
                                extra_jobs = len(colony_tasks) + (
                                    1 if calc_stats and colony_tasks else 0
                                )
                                pipeline.total = (pipeline.total or 0) + extra_jobs
                                pipeline.set_postfix_str(
                                    f"N={signals} colony ready ({len(colony_tasks)} BATS)",
                                    refresh=True,
                                )
                                pipeline.update(1)

                                for task in colony_tasks:
                                    bats_future = executor.submit(run_bats_worker, task)
                                    future_metadata[bats_future] = ("bats", signals)
                                    pending_futures.add(bats_future)

                                if not colony_tasks:
                                    final_results[signals] = None

                                submit_colony(executor)

                            elif task_type == "bats":
                                result = future.result()
                                grouped_results[signals].append(result)
                                remaining = tasks_remaining[signals]
                                if remaining is not None:
                                    tasks_remaining[signals] = remaining - 1
                                pipeline.update(1)
                                launch_stats_if_ready(executor, signals)

                            elif task_type == "stats":
                                final_results[signals] = future.result()
                                pipeline.update(1)

                        except Exception as e:
                            print(
                                f"Task '{task_type}' for {signals} signals failed with error: {e}"
                            )
                            pipeline.update(1)
                            if task_type == "colony":
                                submit_colony(executor)
                            elif task_type == "bats" and tasks_remaining[signals] is not None:
                                remaining = tasks_remaining[signals]
                                if remaining is not None:
                                    tasks_remaining[signals] = remaining - 1
                                launch_stats_if_ready(executor, signals)
        finally:
            pipeline.close()

        out_path = _resolve_output_dir(output_dir)
        _write_outputs(
            out_path,
            self.t,
            self.d,
            final_results,
            f_init_by_n,
            k_init_by_n,
        )

        extras = {
            **sort_extras,
            "output_dir": str(out_path),
            "f_init": f_work,
            "k_init": k_work,
            "prior_n_std": prior_n_std,
        }

        return DraculaResult(
            by_n=final_results,
            min_signals=min_signals,
            max_signals=max_signals,
            extras=extras,
        )
