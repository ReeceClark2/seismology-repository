"""Serial initial-condition grid search (Stage D).

Candidates are found on an (f, ln k) grid. After each acceptance the
cumulative joint model is rebuilt against the original subband record.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from bats import (
    DECAY_FLOOR,
    evaluate_frequency_decay_grid,
    get_model,
    get_statistics,
)
from subbands import SubbandWindow, _typical_psd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


SignalCountMode = Literal["fixed", "automatic"]
SignalCountMethod = Literal["global_likelihood", "peak_contrast"]


@dataclass
class DetectedCandidate:
    subband_index: int
    iteration: int
    frequency: float
    decay_rate: float
    log_decay: float
    log_probability: float
    accepted: bool
    reason: str
    source: str = "detected"
    peak_contrast: float | None = None
    floor_median: float | None = None
    floor_std: float | None = None
    global_likelihood: float | None = None
    min_f: float | None = None
    max_f: float | None = None


@dataclass
class SubbandSearchResult:
    subband_index: int
    min_f: float
    max_f: float
    candidates: list[DetectedCandidate]
    selected: list[DetectedCandidate]
    stopping_reason: str
    extras: dict[str, Any] = field(default_factory=dict)


def estimate_min_decay_rate(t: ArrayLike, d: ArrayLike, threshold: float) -> float:
    """Lower bound on k from A(T)=A(0)exp(-kT), floored above zero."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    duration = float(t[-1] - t[0])
    peak = float(np.max(np.abs(d)))
    if duration <= 0.0:
        raise ValueError("Subband duration must be positive")
    if peak <= 0.0 or threshold <= 0.0:
        return float(DECAY_FLOOR)
    return float(max(DECAY_FLOOR, np.log(peak / threshold) / duration))


def peak_contrast_from_surface(
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """One-sided high-side cleaning of a log-probability surface."""
    values = np.asarray(probabilities, dtype=float)
    finite = np.isfinite(values)
    if int(np.count_nonzero(finite)) < 3:
        return {
            "peak_contrast": np.nan,
            "floor_median": np.nan,
            "floor_std": np.nan,
            "typical": np.zeros_like(values, dtype=bool),
            "atypical": np.zeros_like(values, dtype=bool),
            "reason": "too_few_finite_grid_points",
        }

    flat = values[finite]
    kept, method = _typical_psd(flat)
    floor_median = float(np.median(kept))
    floor_std = float(np.std(kept))
    typical = np.zeros_like(values, dtype=bool)
    # Reconstruct typical mask: finite cells whose value is within the kept set
    # (high-side outliers are atypical).
    high_cut = floor_median + 3.0 * (floor_std if floor_std > 0.0 else 0.0)
    typical[finite] = values[finite] <= high_cut
    atypical = finite & ~typical
    maximum = float(np.nanmax(values))
    if not np.isfinite(floor_std) or floor_std <= 0.0:
        contrast = np.nan
        reason = "zero_or_nonfinite_floor_std"
    else:
        contrast = (maximum - floor_median) / floor_std
        reason = "ok"
    return {
        "peak_contrast": float(contrast) if np.isfinite(contrast) else np.nan,
        "floor_median": floor_median,
        "floor_std": floor_std,
        "typical": typical,
        "atypical": atypical,
        "method": method,
        "reason": reason,
        "maximum": maximum,
    }


def _plot_peak_contrast(
    path: Path,
    f_space: np.ndarray,
    log_k_space: np.ndarray,
    probabilities: np.ndarray,
    contrast: dict[str, Any],
    selected_f: float,
    selected_k: float,
    threshold: float,
    accepted: bool,
) -> None:
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    log_prob = np.ma.masked_invalid(probabilities)
    mesh = axes[0].pcolormesh(
        log_k_space,
        f_space,
        log_prob,
        shading="auto",
        cmap="viridis",
        rasterized=True,
    )
    fig.colorbar(mesh, ax=axes[0], label="Log probability")
    axes[0].scatter(
        [np.log(max(selected_k, DECAY_FLOOR))],
        [selected_f],
        marker="x",
        color="red",
        s=80,
    )
    axes[0].set_xlabel("ln(k)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].set_title("Probability surface")

    classified = np.zeros(probabilities.shape)
    classified[contrast["typical"]] = 1.0
    classified[contrast["atypical"]] = 2.0
    classified[~np.isfinite(probabilities)] = 0.0
    axes[1].pcolormesh(
        log_k_space,
        f_space,
        classified,
        shading="auto",
        cmap="Accent",
        vmin=0,
        vmax=2,
        rasterized=True,
    )
    axes[1].scatter(
        [np.log(max(selected_k, DECAY_FLOOR))],
        [selected_f],
        marker="x",
        color="red",
        s=80,
    )
    axes[1].set_xlabel("ln(k)")
    contrast_value = contrast["peak_contrast"]
    axes[1].set_title(
        f"typical/atypical  contrast={contrast_value:.3g}  "
        f"threshold={threshold:g}  "
        f"{'accept' if accepted else 'reject'}"
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual(
    path: Path,
    t: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, before, color="black", lw=0.7, label="Before")
    ax.plot(t, after, color="C0", lw=0.7, label="After joint residual")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _cumulative_likelihood(
    t: np.ndarray,
    d: np.ndarray,
    frequencies: list[float],
    decays: list[float],
) -> float:
    if not frequencies:
        return float("nan")
    try:
        stats = get_statistics(
            t,
            d,
            frequencies,
            decays,
            calc_p_spec=False,
            calc_cov_mat=False,
            calc_f_unc=False,
            calc_k_unc=False,
        )
        value = float(np.asarray(stats.glob_LL))
    except Exception:
        return float("nan")
    if not np.isfinite(value):
        return float("nan")
    return value


def search_subband(
    window: SubbandWindow,
    *,
    signal_count_mode: SignalCountMode = "fixed",
    signal_count_method: SignalCountMethod = "peak_contrast",
    signals_per_subband: int = 5,
    max_candidates: int = 10,
    peak_contrast_threshold: float = 5.0,
    global_likelihood_patience: int = 2,
    selection_fallback: SignalCountMethod | None = "peak_contrast",
    max_k: float = 1.2e-4,
    f_points: int = 250,
    k_points: int = 250,
    grid_batch_size: int = 4_096,
    output_dir: Path | None = None,
    save_diagnostics: bool = True,
) -> SubbandSearchResult:
    """Sequential (f, ln k) search on one peak-trimmed subband window."""
    if not window.accepted or window.t_local is None or window.d_local is None:
        return SubbandSearchResult(
            subband_index=window.spec.index,
            min_f=window.spec.min_f,
            max_f=window.spec.max_f,
            candidates=[],
            selected=[],
            stopping_reason="subband_not_accepted",
        )

    if signal_count_mode not in ("fixed", "automatic"):
        raise ValueError(
            f"signal_count_mode must be 'fixed' or 'automatic', got {signal_count_mode!r}"
        )
    if signal_count_mode == "automatic" and signal_count_method not in (
        "global_likelihood",
        "peak_contrast",
    ):
        raise ValueError(
            "signal_count_method must be 'global_likelihood' or 'peak_contrast', "
            f"got {signal_count_method!r}"
        )

    t = np.asarray(window.t_local, dtype=float)
    original = np.asarray(window.d_local, dtype=float)
    threshold = float(window.threshold or 0.0)
    min_k = estimate_min_decay_rate(t, original, threshold)
    min_k = min(max(min_k, float(DECAY_FLOOR)), float(max_k) * 0.999)
    hard_max = int(max_candidates)
    if signal_count_mode == "fixed":
        hard_max = min(hard_max, int(signals_per_subband))

    frequencies: list[float] = []
    decays: list[float] = []
    likelihoods: list[float] = []
    candidates: list[DetectedCandidate] = []
    consecutive_ll_drops = 0
    stopping_reason = "reached_candidate_limit"
    method = signal_count_method if signal_count_mode == "automatic" else "fixed"

    for iteration in range(1, hard_max + 1):
        if frequencies:
            cumulative = np.asarray(
                get_model(t, original, frequencies, decays)
            )
            residual = original - cumulative
        else:
            residual = original.copy()
            cumulative = np.zeros_like(original)

        f_space, k_space, log_k_space, surface = evaluate_frequency_decay_grid(
            t,
            residual,
            min_f=window.spec.min_f,
            max_f=window.spec.max_f,
            min_k=min_k,
            max_k=float(max_k),
            f_points=f_points,
            k_points=k_points,
            grid_batch_size=grid_batch_size,
        )
        if not np.any(np.isfinite(surface)):
            stopping_reason = "no_finite_grid_points"
            break

        idx = int(np.nanargmax(surface))
        i_f, i_k = np.unravel_index(idx, surface.shape)
        selected_f = float(f_space[i_f])
        selected_k = float(k_space[i_k])
        selected_lp = float(surface[i_f, i_k])
        contrast = peak_contrast_from_surface(surface)

        accepted = True
        reason = "accepted"
        glob_ll = None

        if signal_count_mode == "automatic" and method == "peak_contrast":
            value = contrast["peak_contrast"]
            if contrast["reason"] != "ok" or not np.isfinite(value):
                accepted = False
                reason = contrast["reason"]
            elif value < peak_contrast_threshold:
                accepted = False
                reason = "below_peak_contrast_threshold"
            if not accepted:
                stopping_reason = reason
                candidate = DetectedCandidate(
                    subband_index=window.spec.index,
                    iteration=iteration,
                    frequency=selected_f,
                    decay_rate=selected_k,
                    log_decay=float(np.log(selected_k)),
                    log_probability=selected_lp,
                    accepted=False,
                    reason=reason,
                    peak_contrast=value if np.isfinite(value) else None,
                    floor_median=contrast["floor_median"],
                    floor_std=contrast["floor_std"],
                    min_f=window.spec.min_f,
                    max_f=window.spec.max_f,
                )
                candidates.append(candidate)
                if save_diagnostics and output_dir is not None:
                    _save_iteration_diagnostics(
                        output_dir,
                        window.spec.index,
                        iteration,
                        t,
                        residual,
                        original - cumulative if frequencies else residual,
                        f_space,
                        log_k_space,
                        surface,
                        contrast,
                        selected_f,
                        selected_k,
                        peak_contrast_threshold,
                        False,
                    )
                break

        if accepted:
            frequencies.append(selected_f)
            decays.append(selected_k)
            if signal_count_mode == "automatic" and method == "global_likelihood":
                glob_ll = _cumulative_likelihood(t, original, frequencies, decays)
                if not np.isfinite(glob_ll):
                    if selection_fallback == "peak_contrast":
                        method = "peak_contrast"
                        glob_ll = None
                        value = contrast["peak_contrast"]
                        if contrast["reason"] != "ok" or not np.isfinite(value):
                            frequencies.pop()
                            decays.pop()
                            accepted = False
                            reason = contrast["reason"]
                        elif value < peak_contrast_threshold:
                            frequencies.pop()
                            decays.pop()
                            accepted = False
                            reason = "below_peak_contrast_threshold"
                    else:
                        frequencies.pop()
                        decays.pop()
                        stopping_reason = "nonfinite_global_likelihood"
                        accepted = False
                        reason = stopping_reason
                else:
                    likelihoods.append(glob_ll)
                    if len(likelihoods) >= 2 and glob_ll < likelihoods[-2]:
                        consecutive_ll_drops += 1
                    else:
                        consecutive_ll_drops = 0
                    if consecutive_ll_drops >= global_likelihood_patience:
                        stopping_reason = "global_likelihood_patience"

        candidate = DetectedCandidate(
            subband_index=window.spec.index,
            iteration=iteration,
            frequency=selected_f,
            decay_rate=selected_k,
            log_decay=float(np.log(max(selected_k, DECAY_FLOOR))),
            log_probability=selected_lp,
            accepted=accepted,
            reason=reason,
            peak_contrast=contrast["peak_contrast"]
            if np.isfinite(contrast["peak_contrast"])
            else None,
            floor_median=contrast["floor_median"],
            floor_std=contrast["floor_std"],
            global_likelihood=glob_ll,
            min_f=window.spec.min_f,
            max_f=window.spec.max_f,
        )
        candidates.append(candidate)

        if save_diagnostics and output_dir is not None:
            after = original - np.asarray(
                get_model(t, original, frequencies, decays)
            ) if frequencies else residual
            _save_iteration_diagnostics(
                output_dir,
                window.spec.index,
                iteration,
                t,
                residual,
                after,
                f_space,
                log_k_space,
                surface,
                contrast,
                selected_f,
                selected_k,
                peak_contrast_threshold,
                accepted,
            )

        if not accepted:
            stopping_reason = reason
            break
        if (
            signal_count_mode == "automatic"
            and method == "global_likelihood"
            and consecutive_ll_drops >= global_likelihood_patience
        ):
            break
        if signal_count_mode == "fixed" and len(frequencies) >= signals_per_subband:
            stopping_reason = "fixed_count_reached"
            break

    selected = [item for item in candidates if item.accepted]
    if (
        signal_count_mode == "automatic"
        and method == "global_likelihood"
        and likelihoods
    ):
        keep = int(np.argmax(likelihoods)) + 1
        kept: list[DetectedCandidate] = []
        n_kept = 0
        for item in candidates:
            if not item.accepted:
                continue
            n_kept += 1
            if n_kept <= keep:
                kept.append(item)
            else:
                item.accepted = False
                item.reason = "beyond_max_global_likelihood"
        selected = kept
        frequencies = frequencies[:keep]
        decays = decays[:keep]

    return SubbandSearchResult(
        subband_index=window.spec.index,
        min_f=window.spec.min_f,
        max_f=window.spec.max_f,
        candidates=candidates,
        selected=selected,
        stopping_reason=stopping_reason,
        extras={
            "min_k": min_k,
            "max_k": float(max_k),
            "signal_count_mode": signal_count_mode,
            "signal_count_method": method,
            "likelihoods": list(likelihoods),
        },
    )


def _save_iteration_diagnostics(
    output_dir: Path,
    subband_index: int,
    iteration: int,
    t: np.ndarray,
    residual_before: np.ndarray,
    residual_after: np.ndarray,
    f_space: np.ndarray,
    log_k_space: np.ndarray,
    surface: np.ndarray,
    contrast: dict[str, Any],
    selected_f: float,
    selected_k: float,
    threshold: float,
    accepted: bool,
) -> None:
    signal_dir = (
        output_dir
        / f"subband_{subband_index:03d}"
        / f"signal_{iteration:03d}"
    )
    signal_dir.mkdir(parents=True, exist_ok=True)
    _plot_peak_contrast(
        signal_dir / "probability_surface.png",
        f_space,
        log_k_space,
        surface,
        contrast,
        selected_f,
        selected_k,
        threshold,
        accepted,
    )
    _plot_residual(
        signal_dir / "residual_before_after.png",
        t,
        residual_before,
        residual_after,
    )
    (signal_dir / "metadata.json").write_text(
        json.dumps(
            {
                "frequency": selected_f,
                "decay_rate": selected_k,
                "log_decay": float(np.log(max(selected_k, DECAY_FLOOR))),
                "peak_contrast": contrast["peak_contrast"],
                "floor_median": contrast["floor_median"],
                "floor_std": contrast["floor_std"],
                "threshold": threshold,
                "accepted": accepted,
            },
            indent=2,
        )
        + "\n"
    )


def save_candidates_csv(path: Path, selected: list[DetectedCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subband_index",
                "signal_index",
                "frequency_hz",
                "decay_rate",
                "log_decay",
                "log_probability",
                "peak_contrast",
                "global_likelihood",
                "source",
                "reason",
            ]
        )
        by_band: dict[int, int] = {}
        for item in selected:
            by_band[item.subband_index] = by_band.get(item.subband_index, 0) + 1
            writer.writerow(
                [
                    item.subband_index,
                    by_band[item.subband_index],
                    item.frequency,
                    item.decay_rate,
                    item.log_decay,
                    item.log_probability,
                    item.peak_contrast,
                    item.global_likelihood,
                    item.source,
                    item.reason,
                ]
            )


def assemble_initial_parameters(
    selected: list[DetectedCandidate],
) -> tuple[np.ndarray, np.ndarray]:
    if not selected:
        return np.zeros(0), np.zeros(0)
    ordered = sorted(selected, key=lambda item: (item.subband_index, item.iteration))
    frequencies = np.asarray([item.frequency for item in ordered], dtype=float)
    decays = np.asarray([item.decay_rate for item in ordered], dtype=float)
    return frequencies, decays


def bandwidths_from_candidates(
    selected: list[DetectedCandidate],
    prior_n_std: float,
    max_k: float,
    decay_floor: float = DECAY_FLOOR,
    default_f_bw: float = 1e-3,
    default_k_bw: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert subband edges into prior scales ``f_bw`` and log-space ``k_bw``."""
    if not selected:
        raise ValueError("bounds_mode='initial_subband' requires detected candidates")
    n_std = float(prior_n_std)
    if n_std <= 0.0:
        raise ValueError(f"prior_n_std must be > 0, got {prior_n_std}")
    ordered = sorted(selected, key=lambda item: (item.subband_index, item.iteration))
    f_bw: list[float] = []
    k_bw: list[float] = []
    n_with_band = 0
    for item in ordered:
        if item.min_f is None or item.max_f is None:
            f_bw.append(float(default_f_bw))
        else:
            n_with_band += 1
            min_f = float(item.min_f)
            max_f = float(item.max_f)
            width = max(max_f - min_f, 1e-12)
            inside = min(item.frequency - min_f, max_f - item.frequency)
            f_half = max(inside, 0.05 * width)
            f_bw.append(f_half / n_std)
        k = max(float(item.decay_rate), float(decay_floor))
        log_k = float(np.log(k))
        log_lo = float(np.log(decay_floor))
        log_hi = float(np.log(max(float(max_k), k * 1.01)))
        k_inside = min(log_k - log_lo, log_hi - log_k)
        k_half = max(k_inside, 1e-3)
        k_bw.append(k_half / n_std)
    if n_with_band == 0:
        raise ValueError(
            "bounds_mode='initial_subband' needs subband frequency limits "
            "on the detected candidates"
        )
    return np.asarray(f_bw, dtype=float), np.asarray(k_bw, dtype=float)


@dataclass
class SubbandSearchJob:
    """Picklable payload for one independent Stage D subband search."""

    window: SubbandWindow
    output_dir: str | None
    signal_count_mode: str
    signal_count_method: str
    signals_per_subband: int
    max_candidates: int
    peak_contrast_threshold: float
    global_likelihood_patience: int
    selection_fallback: str | None
    max_k: float
    f_points: int
    k_points: int
    grid_batch_size: int
    save_diagnostics: bool


def run_subband_search_job(job: SubbandSearchJob) -> SubbandSearchResult:
    output_dir = Path(job.output_dir) if job.output_dir is not None else None
    return search_subband(
        job.window,
        signal_count_mode=job.signal_count_mode,  # type: ignore[arg-type]
        signal_count_method=job.signal_count_method,  # type: ignore[arg-type]
        signals_per_subband=job.signals_per_subband,
        max_candidates=job.max_candidates,
        peak_contrast_threshold=job.peak_contrast_threshold,
        global_likelihood_patience=job.global_likelihood_patience,
        selection_fallback=job.selection_fallback,  # type: ignore[arg-type]
        max_k=job.max_k,
        f_points=job.f_points,
        k_points=job.k_points,
        grid_batch_size=job.grid_batch_size,
        output_dir=output_dir,
        save_diagnostics=job.save_diagnostics,
    )


def dedupe_candidates(
    known_f: np.ndarray,
    known_k: np.ndarray,
    detected: list[DetectedCandidate],
    frequency_separation: float,
) -> list[DetectedCandidate]:
    """Drop detected candidates too close in frequency to supplied ones."""
    kept: list[DetectedCandidate] = []
    known = list(np.asarray(known_f, dtype=float))
    for item in detected:
        if any(abs(item.frequency - freq) < frequency_separation for freq in known):
            item.accepted = False
            item.reason = "duplicate_of_supplied"
            continue
        kept.append(item)
        known.append(item.frequency)
    return kept
