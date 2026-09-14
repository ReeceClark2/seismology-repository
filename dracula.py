from __future__ import annotations

import concurrent.futures
import csv
import json
import multiprocessing
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from tqdm.auto import tqdm

from bats import (
    BATSResult,
    DECAY_FLOOR,
    StatisticsResult,
    as_1d_float,
    broadcast_bandwidth,
    get_model,
    get_statistics,
    prefix_bandwidth,
    rank_by_power,
    resolve_bounds_mode,
    resolve_imposed_surface,
    split_numpyro_kwargs,
)
from colony import (
    ColonyJob,
    init_parallel_worker,
    resolve_progress_mode,
    resolve_signals_per_worker,
    run_bats_worker,
    run_colony_worker,
)
from subbands import (
    detect_subband_windows,
    save_initial_condition_stage,
    slice_to_analysis_interval,
)
from detection import (
    DetectedCandidate,
    SubbandSearchJob,
    assemble_initial_parameters,
    bandwidths_from_candidates,
    dedupe_candidates,
    run_subband_search_job,
    save_candidates_csv,
)

OUTPUT_SUBDIRS = (
    "run_configuration",
    "initial_conditions",
    "summary",
    "sampling",
    "final_results",
)


def _np(value: Any) -> np.ndarray:
    return np.asarray(value)


def _nearest_power(p_spec: ArrayLike, frequencies: ArrayLike) -> np.ndarray:
    f_grid, p_grid = _np(p_spec).T
    frequencies = _np(frequencies).ravel()
    idx = np.array([np.argmin(np.abs(f_grid - f)) for f in frequencies])
    return p_grid[idx]


def _resolve_output_dir(output_dir: str | os.PathLike[str] | None) -> Path:
    path = Path(output_dir) if output_dir is not None else Path.cwd() / "dracula_output"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = path.parent / f"{path.name}_{timestamp}"

    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class RunLayout:
    """Stage directories under a timestamped Dracula output root."""

    root: Path

    @property
    def run_configuration(self) -> Path:
        return self.root / "run_configuration"

    @property
    def initial_conditions(self) -> Path:
        return self.root / "initial_conditions"

    @property
    def summary(self) -> Path:
        return self.root / "summary"

    @property
    def sampling(self) -> Path:
        return self.root / "sampling"

    @property
    def final_results(self) -> Path:
        return self.root / "final_results"


def create_run_layout(
    output_dir: str | os.PathLike[str] | None = None,
) -> RunLayout:
    """Create the timestamped output tree before any processing."""
    root = _resolve_output_dir(output_dir)
    layout = RunLayout(root=root)
    for name in OUTPUT_SUBDIRS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return layout


def _detect_initial_candidates(
    t: Any,
    d: Any,
    layout: RunLayout,
    *,
    min_f: float,
    max_f: float,
    initial_subband_width: float,
    initial_subband_scaling: float,
    ic_min_points: int,
    ic_n_sigma: float,
    ic_filter_order: int,
    ic_fft_points: int,
    signal_count_mode: str,
    signal_count_method: str,
    signals_per_subband: int,
    max_candidates_per_subband: int,
    peak_contrast_threshold: float,
    global_likelihood_patience: int,
    selection_fallback: str | None,
    ic_max_k: float,
    ic_f_points: int,
    ic_k_points: int,
    ic_grid_batch_size: int,
    save_ic_diagnostics: bool,
    progress_mode: str,
    max_workers: int = 1,
) -> tuple[Any, list[Any], Any, Any]:
    print("Stage C: building dynamic subbands and useful time windows.")
    stage = detect_subband_windows(
        t,
        d,
        min_f=min_f,
        max_f=max_f,
        initial_width=initial_subband_width,
        scaling=initial_subband_scaling,
        min_points=ic_min_points,
        n_sigma=ic_n_sigma,
        filter_order=ic_filter_order,
        fft_points=ic_fft_points,
    )
    save_initial_condition_stage(layout.initial_conditions, stage)

    accepted_windows = [window for window in stage.windows if window.accepted]
    selected: list[Any] = []
    job_kwargs = dict(
        output_dir=str(layout.initial_conditions),
        signal_count_mode=signal_count_mode,
        signal_count_method=signal_count_method,
        signals_per_subband=signals_per_subband,
        max_candidates=max_candidates_per_subband,
        peak_contrast_threshold=peak_contrast_threshold,
        global_likelihood_patience=global_likelihood_patience,
        selection_fallback=selection_fallback,
        max_k=ic_max_k,
        f_points=ic_f_points,
        k_points=ic_k_points,
        grid_batch_size=ic_grid_batch_size,
        save_diagnostics=save_ic_diagnostics,
    )
    jobs = [
        SubbandSearchJob(window=window, **job_kwargs)  # type: ignore[arg-type]
        for window in accepted_windows
    ]
    workers = max(1, int(max_workers))
    print(
        "Stage D: "
        f"{'parallel' if workers > 1 and len(jobs) > 1 else 'serial'} "
        "(f, ln k) grid search with joint residuals."
    )

    if workers > 1 and len(jobs) > 1:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=min(workers, len(jobs)),
            mp_context=ctx,
        ) as executor:
            results = list(executor.map(run_subband_search_job, jobs))
    else:
        iterator = jobs
        if progress_mode == "detailed" and jobs:
            iterator = tqdm(jobs, desc="Stage D subbands", leave=False)
        results = [run_subband_search_job(job) for job in iterator]

    results = sorted(results, key=lambda item: item.subband_index)
    for result in results:
        selected.extend(result.selected)
        print(
            f"  subband {result.subband_index:03d}: "
            f"{len(result.selected)} accepted "
            f"({result.stopping_reason})"
        )

    save_candidates_csv(
        layout.initial_conditions / "selected_candidates.csv",
        selected,
    )
    frequencies, decays = assemble_initial_parameters(selected)
    return stage, selected, frequencies, decays


def resolve_signal_counts(
    signals: int | None,
    min_signals: int | None,
    max_signals: int | None,
) -> tuple[int, int]:
    """Return ``(min_signals, max_signals)`` from exact or ranged arguments."""
    if signals is not None:
        if min_signals is not None or max_signals is not None:
            raise ValueError(
                "signals cannot be combined with min_signals or max_signals"
            )
        count = int(signals)
        if count < 1:
            raise ValueError(f"signals must be >= 1, got {count}")
        return count, count

    if min_signals is None or max_signals is None:
        raise ValueError(
            "Provide signals=N or both min_signals and max_signals"
        )

    min_count = int(min_signals)
    max_count = int(max_signals)
    if min_count < 1:
        raise ValueError(f"min_signals must be >= 1, got {min_count}")
    if max_count < min_count:
        raise ValueError(
            f"max_signals ({max_count}) must be >= min_signals ({min_count})"
        )
    return min_count, max_count


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        if isinstance(item, (bool, int, float, str)):
            return item
        return str(item)
    if array.size <= 64:
        return array.tolist()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _write_run_configuration(path: Path, config: dict[str, Any]) -> None:
    serializable = {key: _jsonable(value) for key, value in config.items()}
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")


def _write_signal_csv(path: Path, stats: StatisticsResult) -> None:
    order = np.argsort(_np(stats.fs))
    fs = _np(stats.fs)[order]
    f_unc = _np(stats.f_unc)[order]
    ks = _np(stats.ks)[order]
    k_unc = _np(stats.k_unc)[order]
    log_ks = _np(stats.log_ks)[order] if stats.log_ks is not None else np.log(ks)
    if stats.sigma_log_k is None:
        sigma_log_k = np.full_like(ks, np.nan)
        factor = np.full_like(ks, np.nan)
        k_lower = np.full_like(ks, np.nan)
        k_upper = np.full_like(ks, np.nan)
    else:
        sigma_log_k = _np(stats.sigma_log_k)[order]
        factor = _np(stats.k_uncertainty_factor)[order]
        k_lower = _np(stats.k_lower)[order]
        k_upper = _np(stats.k_upper)[order]

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frequencies",
                "frequency_uncertainties",
                "decay_rates",
                "log_decay_rates",
                "log_decay_uncertainties",
                "decay_uncertainty_factor",
                "decay_rate_lower",
                "decay_rate_upper",
                "decay_rate_uncertainties",
            ]
        )
        for row in zip(
            fs,
            f_unc,
            ks,
            log_ks,
            sigma_log_k,
            factor,
            k_lower,
            k_upper,
            k_unc,
        ):
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


def _plot_fk(
    path: Path,
    f_init: ArrayLike,
    k_init: ArrayLike,
    stats: StatisticsResult,
    subband_edges: list[float] | None = None,
) -> None:
    f0 = _np(f_init).ravel()
    k0 = _np(k_init).ravel()
    f1 = _np(stats.fs).ravel()
    log_k0 = np.log(np.maximum(k0, float(DECAY_FLOOR)))
    if stats.log_ks is None:
        log_k1 = np.log(np.maximum(_np(stats.ks).ravel(), float(DECAY_FLOOR)))
    else:
        log_k1 = _np(stats.log_ks).ravel()
    f_unc = _np(stats.f_unc).ravel() if stats.f_unc is not None else None
    sig = (
        _np(stats.sigma_log_k).ravel()
        if stats.sigma_log_k is not None
        else None
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    if subband_edges:
        for edge in subband_edges:
            ax.axvline(float(edge), color="0.8", lw=0.6, zorder=0)
    ax.scatter(f0, log_k0, s=36, color="black", label="Initial (f, ln k)", zorder=3)
    ax.errorbar(
        f1,
        log_k1,
        xerr=f_unc if f_unc is not None and f_unc.size == f1.size else None,
        yerr=sig if sig is not None and sig.size == f1.size else None,
        fmt="x",
        color="C3",
        ms=8,
        elinewidth=0.9,
        capsize=2,
        label="Sampled (f, ln k)",
        zorder=4,
    )
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("ln(k)")
    ax.set_title("Initial vs sampled frequency–decay")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_covariance(path: Path, stats: StatisticsResult) -> None:
    cov = _np(stats.cov_mat)
    if cov.ndim != 2 or cov.size == 0:
        return
    n = int(cov.shape[0] // 2)
    labels = [f"f{i + 1}" for i in range(n)] + [f"ln k{i + 1}" for i in range(n)]
    fig, ax = plt.subplots(figsize=(8, 7))
    mesh = ax.imshow(cov, cmap="coolwarm", interpolation="nearest")
    fig.colorbar(mesh, ax=ax, label="Covariance")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title("Hessian covariance [f, ln k]")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_best_n(
    by_n: dict[int, StatisticsResult | BATSResult | None],
) -> int | None:
    best_n: int | None = None
    best_ll = -float("inf")
    for n, result in by_n.items():
        if not isinstance(result, StatisticsResult):
            continue
        value = float(np.asarray(result.glob_LL))
        if np.isfinite(value) and value > best_ll:
            best_ll = value
            best_n = int(n)
    return best_n


def _write_chain_diagnostics(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "n_signals",
        "chunk_seed",
        "chain",
        "seed",
        "freq_lo",
        "freq_hi",
        "min_potential_energy",
        "n_finite_samples",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _write_final_results(
    output_dir: Path,
    t: ArrayLike,
    d: ArrayLike,
    by_n: dict[int, StatisticsResult | BATSResult | None],
    f_init_by_n: dict[int, Any],
    k_init_by_n: dict[int, Any],
    subband_edges: list[float] | None = None,
) -> int | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_n = _select_best_n(by_n)
    if selected_n is None:
        return None
    result = by_n[selected_n]
    if not isinstance(result, StatisticsResult):
        return None
    _write_signal_csv(output_dir / "selected_signals.csv", result)
    _plot_timeseries(
        output_dir / "selected_timeseries.png",
        t,
        d,
        result.fs,
        result.ks,
        result.SNR,
        result.variance,
        selected_n,
    )
    before = get_statistics(t, d, f_init_by_n[selected_n], k_init_by_n[selected_n])
    _plot_power_spectrum(
        output_dir / "selected_power_spectrum.png",
        before,
        result,
        f_init_by_n[selected_n],
        selected_n,
    )
    _plot_fk(
        output_dir / "selected_fk.png",
        f_init_by_n[selected_n],
        k_init_by_n[selected_n],
        result,
        subband_edges=subband_edges,
    )
    _plot_covariance(output_dir / "selected_covariance.png", result)
    np.save(output_dir / "selected_covariance.npy", _np(result.cov_mat))
    residual = _np(d) - _np(get_model(t, d, result.fs, result.ks))
    np.savetxt(
        output_dir / "selected_residual.csv",
        np.column_stack([_np(t).ravel(), residual.ravel()]),
        delimiter=",",
        header="time_s,residual",
        comments="",
    )
    (output_dir / "selected_model.json").write_text(
        json.dumps(
            {
                "N": int(selected_n),
                "SNR": float(result.SNR),
                "variance": float(result.variance),
                "glob_LL": float(result.glob_LL),
            },
            indent=2,
        )
        + "\n"
    )
    return selected_n


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
        f_init: ArrayLike | None = None,
        k_init: ArrayLike | None = None,
    ) -> None:
        self.t = as_1d_float(t, "t")
        self.d = as_1d_float(d, "d")
        if self.t.shape[0] != self.d.shape[0]:
            raise ValueError(
                f"t and d must have the same length, got {self.t.shape[0]} and {self.d.shape[0]}"
            )
        if int(self.t.shape[0]) < 2:
            raise ValueError("t and d must contain at least two samples")
        if not bool(np.all(np.isfinite(_np(self.t)))) or not bool(
            np.all(np.isfinite(_np(self.d)))
        ):
            raise ValueError("t and d must be finite")
        if not bool(np.all(np.diff(_np(self.t)) > 0.0)):
            raise ValueError("t must be strictly increasing")

        if (f_init is None) != (k_init is None):
            raise ValueError(
                "f_init and k_init must both be supplied or both be None"
            )

        if f_init is None:
            self.f_init = None
            self.k_init = None
            return

        self.f_init = as_1d_float(f_init, "f_init")
        self.k_init = as_1d_float(k_init, "k_init")
        if self.f_init.shape[0] != self.k_init.shape[0]:
            raise ValueError(
                "f_init and k_init must have the same length, "
                f"got {self.f_init.shape[0]} and {self.k_init.shape[0]}"
            )
        if not bool(np.all(np.isfinite(_np(self.f_init)))):
            raise ValueError("f_init must be finite")
        if not bool(np.all(np.isfinite(_np(self.k_init)))):
            raise ValueError("k_init must be finite")
        if not bool(np.all(_np(self.k_init) > 0.0)):
            raise ValueError("k_init must be strictly positive")

    def _require_initial_conditions(self) -> tuple[Any, Any]:
        if self.f_init is None or self.k_init is None:
            raise RuntimeError("Pass both f_init and k_init to Dracula.")
        return self.f_init, self.k_init

    def _prepare_signals(
        self,
        f_bw: float | ArrayLike | None,
        k_bw: float | ArrayLike | None,
        sort_signals: bool,
        f_init: ArrayLike | None = None,
        k_init: ArrayLike | None = None,
        t: ArrayLike | None = None,
        d: ArrayLike | None = None,
    ) -> tuple[Any, Any, float | Any | None, float | Any | None, dict[str, Any]]:
        f_work = self.f_init if f_init is None else as_1d_float(f_init, "f_init")
        k_work = self.k_init if k_init is None else as_1d_float(k_init, "k_init")
        t_work = self.t if t is None else as_1d_float(t, "t")
        d_work = self.d if d is None else as_1d_float(d, "d")
        f_bw_work = f_bw
        k_bw_work = k_bw
        extras: dict[str, Any] = {"sort_signals": sort_signals}

        if f_work is None or k_work is None:
            raise RuntimeError("Pass both f_init and k_init to Dracula.")

        if sort_signals:
            order, powers, _ = rank_by_power(t_work, d_work, f_work, k_work)
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
        signals_per_worker: int | None = None,
        min_signals: int | None = None,
        max_signals: int | None = None,
        f_bw: float | ArrayLike | None = None,
        k_bw: float | ArrayLike | None = None,
        W: int = 1_000,
        S: int = 2_000,
        calc_stats: bool = True,
        stats_at_end: bool = False,
        max_cores: int | None = None,
        sort_signals: bool = True,
        output_dir: str | os.PathLike[str] | None = None,
        prior_n_std: float = 5.0,
        unbounded: bool | None = None,
        signals: int | None = None,
        f_per_worker: int | None = None,
        progress_mode: str = "detailed",
        imposed_surface: str | None = None,
        bounds_mode: str = "bandwidth",
        min_f: float | None = None,
        max_f: float | None = None,
        initial_subband_width: float | None = None,
        initial_subband_scaling: float = 4.0,
        ic_min_points: int = 30,
        ic_n_sigma: float = 3.0,
        ic_filter_order: int = 4,
        ic_fft_points: int = 2000,
        signal_count_mode: str = "fixed",
        signal_count_method: str = "peak_contrast",
        signals_per_subband: int = 5,
        max_candidates_per_subband: int = 10,
        peak_contrast_threshold: float = 5.0,
        global_likelihood_patience: int = 2,
        selection_fallback: str | None = "peak_contrast",
        ic_max_k: float = 1.2e-4,
        ic_f_points: int = 250,
        ic_k_points: int = 250,
        ic_grid_batch_size: int = 4_096,
        ic_frequency_separation: float | None = None,
        save_ic_diagnostics: bool = True,
        n_chains: int = 1,
        **kwargs: Any,
    ) -> DraculaResult:
        progress_mode = resolve_progress_mode(progress_mode)
        if prior_n_std <= 0:
            raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")
        n_chains = max(1, int(n_chains))
        if max_cores is None:
            max_cores = max(1, (os.cpu_count() or 4) - 2)
        worker_count = resolve_signals_per_worker(
            signals_per_worker=signals_per_worker,
            f_per_worker=f_per_worker,
        )
        surface = resolve_imposed_surface(
            imposed_surface=imposed_surface,
            unbounded=unbounded,
        )
        bounds_mode = resolve_bounds_mode(bounds_mode)

        ic_stage = None
        selected_candidates: list[DetectedCandidate] = []
        sample_t = self.t
        sample_d = self.d
        layout: RunLayout | None = None

        detect_kw = dict(
            min_f=min_f,
            max_f=max_f,
            initial_subband_width=initial_subband_width,
            initial_subband_scaling=initial_subband_scaling,
            ic_min_points=ic_min_points,
            ic_n_sigma=ic_n_sigma,
            ic_filter_order=ic_filter_order,
            ic_fft_points=ic_fft_points,
            signal_count_mode=signal_count_mode,
            signal_count_method=signal_count_method,
            signals_per_subband=signals_per_subband,
            max_candidates_per_subband=max_candidates_per_subband,
            peak_contrast_threshold=peak_contrast_threshold,
            global_likelihood_patience=global_likelihood_patience,
            selection_fallback=selection_fallback,
            ic_max_k=ic_max_k,
            ic_f_points=ic_f_points,
            ic_k_points=ic_k_points,
            ic_grid_batch_size=ic_grid_batch_size,
            save_ic_diagnostics=save_ic_diagnostics,
            progress_mode=progress_mode,
            max_workers=max_cores,
        )

        if self.f_init is None or self.k_init is None:
            if min_f is None or max_f is None or initial_subband_width is None:
                raise ValueError(
                    "Without f_init and k_init, dispatch requires min_f, "
                    "max_f, and initial_subband_width"
                )
            layout = create_run_layout(output_dir)
            ic_stage, selected_candidates, frequencies, decays = (
                _detect_initial_candidates(
                    self.t,
                    self.d,
                    layout,
                    **detect_kw,  # type: ignore[arg-type]
                )
            )
            n_total = int(np.asarray(frequencies).size)
            if n_total < 1:
                raise ValueError("Stage D found no candidate signals")
            f_init = frequencies
            k_init = decays
            sample_t, sample_d = slice_to_analysis_interval(
                self.t, self.d, ic_stage.interval
            )
            print(
                f"Stage D complete: {n_total} candidate signals. "
                "Stage E: Colony/BATS sampling."
            )
            if signals is None and min_signals is None and max_signals is None:
                min_signals, max_signals = n_total, n_total
            else:
                min_signals, max_signals = resolve_signal_counts(
                    signals=signals,
                    min_signals=min_signals,
                    max_signals=max_signals,
                )
            if max_signals > n_total:
                if min_signals > n_total:
                    raise ValueError(
                        f"min_signals ({min_signals}) exceeds detected "
                        f"signals ({n_total})"
                    )
                print(
                    f"Detected {n_total} signals; clamping max_signals "
                    f"from {max_signals} to {n_total}."
                )
                max_signals = n_total
        else:
            f_init, k_init = self._require_initial_conditions()
            n_total = int(f_init.shape[0])
            min_signals, max_signals = resolve_signal_counts(
                signals=signals,
                min_signals=min_signals,
                max_signals=max_signals,
            )
            need_residual = max_signals > n_total
            if need_residual and (
                min_f is None or max_f is None or initial_subband_width is None
            ):
                raise ValueError(
                    f"max_signals ({max_signals}) exceeds supplied ICs ({n_total}); "
                    "pass min_f, max_f, and initial_subband_width to search the residual"
                )
            if bounds_mode == "initial_subband" and not need_residual:
                raise ValueError(
                    "bounds_mode='initial_subband' requires Stage D candidates"
                )
            layout = create_run_layout(output_dir)
            if need_residual:
                residual = np.asarray(self.d, dtype=float) - np.asarray(
                    get_model(self.t, self.d, f_init, k_init)
                )
                print(
                    f"Stage D: {n_total} supplied ICs < max_signals={max_signals}; "
                    "searching the joint residual."
                )
                ic_stage, residual_selected, _, _ = _detect_initial_candidates(
                    self.t,
                    residual,
                    layout,
                    **detect_kw,  # type: ignore[arg-type]
                )
                separation = (
                    float(ic_frequency_separation)
                    if ic_frequency_separation is not None
                    else 0.25 * float(initial_subband_width)
                )
                selected_candidates = dedupe_candidates(
                    np.asarray(f_init),
                    np.asarray(k_init),
                    residual_selected,
                    frequency_separation=separation,
                )
                extra_f, extra_k = assemble_initial_parameters(selected_candidates)
                if extra_f.size:
                    f_init = jnp.concatenate(
                        [jnp.asarray(f_init), jnp.asarray(extra_f)]
                    )
                    k_init = jnp.concatenate(
                        [jnp.asarray(k_init), jnp.asarray(extra_k)]
                    )
                n_total = int(np.asarray(f_init).shape[0])
                supplied = [
                    DetectedCandidate(
                        subband_index=0,
                        iteration=index,
                        frequency=float(freq),
                        decay_rate=float(decay),
                        log_decay=float(
                            np.log(max(float(decay), float(DECAY_FLOOR)))
                        ),
                        log_probability=float("nan"),
                        accepted=True,
                        reason="user_supplied",
                        source="supplied",
                    )
                    for index, (freq, decay) in enumerate(
                        zip(
                            np.asarray(self.f_init).ravel(),
                            np.asarray(self.k_init).ravel(),
                        ),
                        start=1,
                    )
                ]
                save_candidates_csv(
                    layout.initial_conditions / "selected_candidates.csv",
                    supplied + selected_candidates,
                )
                if ic_stage is not None:
                    sample_t, sample_d = slice_to_analysis_interval(
                        self.t, self.d, ic_stage.interval
                    )
                if max_signals > n_total:
                    if min_signals > n_total:
                        raise ValueError(
                            f"min_signals ({min_signals}) exceeds signals available "
                            f"after residual search ({n_total})"
                        )
                    print(
                        f"Residual search produced {n_total} total signals; "
                        f"clamping max_signals from {max_signals} to {n_total}."
                    )
                    max_signals = n_total

        assert layout is not None
        if bounds_mode == "initial_subband":
            if not selected_candidates:
                raise ValueError(
                    "bounds_mode='initial_subband' requires Stage D candidates"
                )
            f_bw, k_bw = bandwidths_from_candidates(
                selected_candidates,
                prior_n_std=prior_n_std,
                max_k=ic_max_k,
            )

        f_work, k_work, f_bw_work, k_bw_work, sort_extras = self._prepare_signals(
            f_bw,
            k_bw,
            sort_signals,
            f_init=f_init,
            k_init=k_init,
            t=sample_t,
            d=sample_d,
        )

        print(f"Limiting execution to {max_cores} concurrent workers.")

        nuts_kwargs, mcmc_kwargs, run_kwargs = split_numpyro_kwargs(kwargs)

        run_config = {
            "signals_per_worker": worker_count,
            "signals": signals,
            "min_signals": min_signals,
            "max_signals": max_signals,
            "W": W,
            "S": S,
            "calc_stats": calc_stats,
            "stats_at_end": stats_at_end,
            "max_cores": max_cores,
            "sort_signals": sort_signals,
            "prior_n_std": prior_n_std,
            "unbounded": unbounded,
            "imposed_surface": surface,
            "bounds_mode": bounds_mode,
            "progress_mode": progress_mode,
            "n_chains": n_chains,
            "n_times": int(np.asarray(sample_t).shape[0]),
            "n_initial_signals": n_total,
            "output_dir": str(layout.root),
            "catalog_free": bool(self.f_init is None),
            "min_f": min_f,
            "max_f": max_f,
        }
        _write_run_configuration(
            layout.run_configuration / "run_config.json",
            run_config,
        )

        grouped_results: dict[int, list[BATSResult]] = {
            s: [] for s in range(min_signals, max_signals + 1)
        }
        tasks_remaining: dict[int, int | None] = {
            s: None for s in range(min_signals, max_signals + 1)
        }
        final_results: dict[int, StatisticsResult | BATSResult | None] = {
            s: None for s in range(min_signals, max_signals + 1)
        }
        deferred_statistics: dict[int, BATSResult] = {}
        f_init_by_n = {s: f_work[:s] for s in range(min_signals, max_signals + 1)}
        k_init_by_n = {s: k_work[:s] for s in range(min_signals, max_signals + 1)}

        n_models = max_signals - min_signals + 1
        colony_queue = list(range(min_signals, max_signals + 1))

        future_metadata: dict[Any, tuple[str, int]] = {}
        pending_futures: set[concurrent.futures.Future[Any]] = set()
        chain_records: list[dict[str, Any]] = []

        def submit_colony(executor: concurrent.futures.ProcessPoolExecutor) -> None:
            if not colony_queue:
                return
            signals = colony_queue.pop(0)
            job = ColonyJob(
                t=sample_t,
                d=sample_d,
                f_init=f_work[:signals],
                k_init=k_work[:signals],
                signals_per_worker=worker_count,
                f_bw=prefix_bandwidth(f_bw_work, signals, "f_bw"),
                k_bw=prefix_bandwidth(k_bw_work, signals, "k_bw"),
                W=W,
                S=S,
                n_signals=signals,
                prior_n_std=prior_n_std,
                imposed_surface=surface,
                progress_mode=progress_mode,
                n_chains=n_chains,
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
            """Combine completed BATS jobs and schedule or defer statistics."""
            if tasks_remaining[signals] != 0:
                return

            if not grouped_results[signals]:
                final_results[signals] = None
                return

            colony_res = sorted(
                grouped_results[signals],
                key=lambda result: (
                    result.seed if result.seed is not None else 0
                ),
            )

            flat_fs = jnp.concatenate(
                [result.fs for result in colony_res]
            )
            flat_ks = jnp.concatenate(
                [result.ks for result in colony_res]
            )

            flat_log_ks = jnp.concatenate(
                [
                    result.log_ks
                    if result.log_ks is not None
                    else jnp.log(result.ks)
                    for result in colony_res
                ]
            )

            combined = BATSResult(
                fs=flat_fs,
                ks=flat_ks,
                log_ks=flat_log_ks,
            )

            # Release the individual BATS results once they have been combined.
            grouped_results[signals].clear()

            if not calc_stats:
                final_results[signals] = combined
                return

            if stats_at_end:
                # Only retain the small frequency/decay arrays. Statistics will run
                # after the process pool has shut down and released worker memory.
                deferred_statistics[signals] = combined
                return

            stats_future = executor.submit(
                get_statistics,
                sample_t,
                sample_d,
                combined.fs,
                combined.ks,
            )

            future_metadata[stats_future] = ("stats", signals)
            pending_futures.add(stats_future)

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
            disable=progress_mode == "none",
        )

        try:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_cores,
                mp_context=ctx,
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
                                    1
                                    if calc_stats and colony_tasks
                                    else 0
                                )
                                pipeline.total = (
                                    pipeline.total or 0
                                ) + extra_jobs

                                pipeline.set_postfix_str(
                                    f"N={signals} colony ready "
                                    f"({len(colony_tasks)} BATS)",
                                    refresh=True,
                                )
                                pipeline.update(1)

                                for task in colony_tasks:
                                    bats_future = executor.submit(
                                        run_bats_worker,
                                        task,
                                    )
                                    future_metadata[bats_future] = (
                                        "bats",
                                        signals,
                                    )
                                    pending_futures.add(bats_future)

                                if not colony_tasks:
                                    final_results[signals] = None

                                submit_colony(executor)

                            elif task_type == "bats":
                                result = future.result()
                                grouped_results[signals].append(result)
                                for row in result.extras.get("chain_diagnostics", []):
                                    chain_records.append(
                                        {"n_signals": int(signals), **row}
                                    )

                                remaining = tasks_remaining[signals]
                                if remaining is not None:
                                    tasks_remaining[signals] = remaining - 1

                                pipeline.update(1)
                                launch_stats_if_ready(executor, signals)

                            elif task_type == "stats":
                                final_results[signals] = future.result()
                                pipeline.update(1)

                        except Exception as error:
                            print(
                                f"Task {task_type!r} for {signals} "
                                f"signals failed: {error}"
                            )
                            pipeline.update(1)

                            if task_type == "colony":
                                submit_colony(executor)

                            elif (
                                task_type == "bats"
                                and tasks_remaining[signals] is not None
                            ):
                                tasks_remaining[signals] -= 1
                                launch_stats_if_ready(
                                    executor,
                                    signals,
                                )

            # The executor has now shut down. Its worker processes and their
            # memory allocations are gone before statistics calculations begin.
            if calc_stats and stats_at_end:
                for signals in sorted(deferred_statistics):
                    combined = deferred_statistics[signals]

                    pipeline.set_postfix_str(
                        f"N={signals} sequential statistics",
                        refresh=True,
                    )

                    try:
                        stats = get_statistics(
                            sample_t,
                            sample_d,
                            combined.fs,
                            combined.ks,
                        )

                        # Ensure all result arrays finish before launching the next job.
                        jax.block_until_ready(
                            (
                                stats.log_prob,
                                stats.variance,
                                stats.SNR,
                                stats.p_spec,
                                stats.glob_LL,
                                stats.cov_mat,
                                stats.f_unc,
                                stats.k_unc,
                                stats.log_ks,
                                stats.sigma_log_k,
                                stats.k_uncertainty_factor,
                                stats.k_lower,
                                stats.k_upper,
                            )
                        )

                        final_results[signals] = stats

                    except Exception as error:
                        final_results[signals] = None
                        print(
                            f"Sequential statistics for N={signals} "
                            f"failed: {error}"
                        )
                    finally:
                        deferred_statistics.pop(signals, None)
                        pipeline.update(1)

        finally:
            pipeline.close()

        _write_outputs(
            layout.sampling,
            sample_t,
            sample_d,
            final_results,
            f_init_by_n,
            k_init_by_n,
        )
        _write_chain_diagnostics(
            layout.sampling / "chain_diagnostics.csv",
            chain_records,
        )
        subband_edges: list[float] | None = None
        if ic_stage is not None:
            edges: list[float] = []
            for spec in ic_stage.specs:
                edges.append(float(spec.min_f))
                edges.append(float(spec.max_f))
            subband_edges = sorted(set(edges))
        selected_n = _write_final_results(
            layout.final_results,
            sample_t,
            sample_d,
            final_results,
            f_init_by_n,
            k_init_by_n,
            subband_edges=subband_edges,
        )

        extras = {
            **sort_extras,
            "output_dir": str(layout.root),
            "sampling_dir": str(layout.sampling),
            "final_results_dir": str(layout.final_results),
            "run_configuration_dir": str(layout.run_configuration),
            "f_init": f_work,
            "k_init": k_work,
            "prior_n_std": prior_n_std,
            "unbounded": unbounded,
            "imposed_surface": surface,
            "bounds_mode": bounds_mode,
            "stats_at_end": bool(stats_at_end),
            "progress_mode": progress_mode,
            "signals_per_worker": worker_count,
            "n_chains": n_chains,
            "sampling_performed": True,
            "selected_n": selected_n,
        }
        if ic_stage is not None:
            extras["initial_condition_stage"] = ic_stage
            extras["detected_candidates"] = selected_candidates
            extras["analysis_interval"] = ic_stage.interval
            extras["initial_conditions_dir"] = str(layout.initial_conditions)

        return DraculaResult(
            by_n=final_results,
            min_signals=min_signals,
            max_signals=max_signals,
            extras=extras,
        )
