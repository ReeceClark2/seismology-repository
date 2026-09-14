"""Dynamic initial-condition subbands and useful-time detection.

Stage C scientific core. Grid-search candidate finding is Stage D.
Waveform download stays in ``initial_conditions.observed_data``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import butter, sosfiltfilt

try:
    import rcrpy
except ImportError:
    rcrpy = None


NoiseMethod = Literal["rcr_ls_mode_68", "median_mad"]


@dataclass
class SubbandSpec:
    """One non-overlapping core frequency interval."""

    index: int
    min_f: float
    max_f: float

    @property
    def width(self) -> float:
        return float(self.max_f - self.min_f)


@dataclass
class SubbandWindow:
    """Useful-time result for one initial-condition subband."""

    spec: SubbandSpec
    accepted: bool
    noise_floor: float | None = None
    threshold: float | None = None
    noise_method: str | None = None
    run_start_index: int | None = None
    run_end_index: int | None = None
    run_start_time: float | None = None
    run_end_time: float | None = None
    peak_index: int | None = None
    peak_time: float | None = None
    peak_amplitude: float | None = None
    trimmed_start_time: float | None = None
    trimmed_end_time: float | None = None
    n_points: int = 0
    t_local: np.ndarray | None = None
    d_local: np.ndarray | None = None
    reason: str = ""
    filter_family: str = "butterworth"
    filter_order: int = 4
    filter_zero_phase: bool = True


@dataclass
class AnalysisInterval:
    """Shared analysis window for the full Dracula run."""

    start_time: float
    end_time: float
    duration: float
    strategy: str
    n_accepted_subbands: int


@dataclass
class InitialConditionStage:
    specs: list[SubbandSpec]
    windows: list[SubbandWindow]
    interval: AnalysisInterval | None
    extras: dict[str, Any] = field(default_factory=dict)


def build_dynamic_subbands(
    min_f: float,
    max_f: float,
    initial_width: float,
    scaling: float,
) -> list[SubbandSpec]:
    """Build contiguous, monotonically narrowing core subbands.

    Width interpolates linearly from ``initial_width`` at ``min_f`` to
    ``initial_width / scaling`` at ``max_f``. The last edge is clamped
    exactly to ``max_f``. Cores do not overlap and have no gaps.
    """
    min_f = float(min_f)
    max_f = float(max_f)
    initial_width = float(initial_width)
    scaling = float(scaling)

    if not np.isfinite(min_f) or not np.isfinite(max_f):
        raise ValueError("min_f and max_f must be finite")
    if min_f <= 0.0:
        raise ValueError(f"min_f must be positive, got {min_f}")
    if max_f <= min_f:
        raise ValueError(f"max_f ({max_f}) must be greater than min_f ({min_f})")
    if not np.isfinite(initial_width) or initial_width <= 0.0:
        raise ValueError(f"initial_subband_width must be > 0, got {initial_width}")
    if not np.isfinite(scaling) or scaling < 1.0:
        raise ValueError(f"initial_subband_scaling must be >= 1, got {scaling}")

    span = max_f - min_f
    final_width = initial_width / scaling
    if initial_width >= span:
        return [SubbandSpec(index=1, min_f=min_f, max_f=max_f)]

    edges = [min_f]
    left = min_f
    for _ in range(1_000_000):
        frac = (left - min_f) / span
        frac = min(max(frac, 0.0), 1.0)
        width = initial_width + (final_width - initial_width) * frac
        nxt = left + width
        remaining = max_f - left
        if nxt >= max_f or remaining <= width * 1.05:
            edges.append(max_f)
            break
        edges.append(nxt)
        left = nxt
    else:
        raise RuntimeError("Dynamic subband construction failed to terminate")

    edges[-1] = max_f
    cleaned = [edges[0]]
    for edge in edges[1:]:
        if edge > cleaned[-1]:
            cleaned.append(float(edge))
    if cleaned[-1] != max_f:
        cleaned[-1] = max_f
    if len(cleaned) < 2:
        return [SubbandSpec(index=1, min_f=min_f, max_f=max_f)]

    last_width = cleaned[-1] - cleaned[-2]
    if len(cleaned) >= 3 and last_width < 0.25 * final_width:
        cleaned.pop(-2)
        cleaned[-1] = max_f

    return [
        SubbandSpec(index=i + 1, min_f=float(lo), max_f=float(hi))
        for i, (lo, hi) in enumerate(zip(cleaned[:-1], cleaned[1:]))
    ]


def bandpass_subband(
    t: ArrayLike,
    d: ArrayLike,
    min_f: float,
    max_f: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass for one initial-condition subband."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    if t.size != d.size:
        raise ValueError("t and d must have the same length")
    if t.size < 2:
        raise ValueError("At least two samples are required")

    dt_values = np.diff(t)
    if np.any(dt_values <= 0):
        raise ValueError("t must be strictly increasing")

    sampling_rate = 1.0 / float(np.median(dt_values))
    nyquist = sampling_rate / 2.0

    if min_f <= 0.0:
        raise ValueError(f"min_f must be positive, got {min_f}")
    if max_f >= nyquist:
        raise ValueError(
            f"max_f ({max_f}) must be below Nyquist ({nyquist})"
        )
    if min_f >= max_f:
        raise ValueError(f"min_f ({min_f}) must be below max_f ({max_f})")

    sos = butter(
        order,
        [min_f, max_f],
        btype="bandpass",
        fs=sampling_rate,
        output="sos",
    )
    return sosfiltfilt(sos, d)


def _typical_psd(psd_band: np.ndarray) -> tuple[np.ndarray, NoiseMethod]:
    if rcrpy is not None:
        rejection = rcrpy.RCR(rcrpy.RejectionTech.LS_MODE_68)
        rejection.perform_rejection(psd_band.tolist())
        flags = np.asarray(rejection.result.flags, dtype=bool)
        kept = psd_band[flags]
        if kept.size == 0:
            raise RuntimeError("RCR rejected every PSD value")
        return kept, "rcr_ls_mode_68"

    median = float(np.median(psd_band))
    mad = float(np.median(np.abs(psd_band - median)))
    scale = 1.4826 * mad
    if scale <= 0.0:
        return psd_band, "median_mad"
    kept = psd_band[psd_band <= median + 3.0 * scale]
    if kept.size < 3:
        kept = psd_band
    return kept, "median_mad"


def get_noise_floor(
    t: ArrayLike,
    d: ArrayLike,
    min_f: float,
    max_f: float,
    fft_points: int = 2000,
) -> tuple[float, float, NoiseMethod]:
    """Band-limited RMS noise and three-sigma threshold from a cleaned PSD."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    if t.size != d.size:
        raise ValueError("t and d must have the same length")
    if t.size < 2:
        raise ValueError("At least two samples are required")
    if fft_points < 2:
        raise ValueError("fft_points must be at least 2")

    dt_values = np.diff(t)
    if np.any(dt_values <= 0):
        raise ValueError("t must be strictly increasing")

    dt = float(np.median(dt_values))
    sampling_rate = 1.0 / dt
    nyquist = sampling_rate / 2.0

    if not 0.0 <= min_f < max_f <= nyquist:
        raise ValueError(
            f"Expected 0 <= min_f < max_f <= {nyquist}, "
            f"got {min_f} and {max_f}"
        )

    desired_df = (max_f - min_f) / (fft_points - 1)
    required_fft_length = int(np.ceil(sampling_rate / desired_df))
    fft_length = max(d.size, required_fft_length)
    fft_length = 1 << (fft_length - 1).bit_length()

    window = np.hanning(d.size)
    window_power = np.sum(window**2)
    fft_values = np.fft.rfft(d * window, n=fft_length)
    frequencies = np.fft.rfftfreq(fft_length, d=dt)
    psd = np.abs(fft_values) ** 2 / (sampling_rate * window_power)

    if fft_length % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    band_mask = (frequencies >= min_f) & (frequencies <= max_f) & np.isfinite(psd)
    psd_band = psd[band_mask]
    if psd_band.size < 3:
        raise ValueError("Too few PSD values were found in the requested subband")

    kept_psd, method = _typical_psd(psd_band)
    mean_psd = 1.44 * float(np.median(kept_psd))
    rms_noise = float(np.sqrt(mean_psd * (max_f - min_f)))
    return rms_noise, 3.0 * rms_noise, method


def find_runs(
    t: ArrayLike,
    d: ArrayLike,
    noise_floor: float,
    n_points: int = 20,
    n_sigma: float = 3.0,
) -> list[dict[str, float | int]]:
    """Find runs terminated by ``n_points`` consecutive samples below threshold."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    if t.size != d.size:
        raise ValueError("t and d must have the same length")
    if n_points < 1:
        raise ValueError("n_points must be at least 1")

    threshold = n_sigma * noise_floor
    above_threshold = np.abs(d) >= threshold

    runs: list[dict[str, float | int]] = []
    start = None
    below_count = 0

    for index, is_above in enumerate(above_threshold):
        if start is None:
            if is_above:
                start = index
                below_count = 0
            continue

        if is_above:
            below_count = 0
        else:
            below_count += 1
            if below_count >= n_points:
                quiet_start = index - n_points + 1
                end = quiet_start - 1
                if end >= start:
                    runs.append(
                        {
                            "start_index": int(start),
                            "end_index": int(end),
                            "length": int(end - start + 1),
                            "start_time": float(t[start]),
                            "end_time": float(t[end]),
                        }
                    )
                start = None
                below_count = 0

    if start is not None:
        end = t.size - 1
        runs.append(
            {
                "start_index": int(start),
                "end_index": int(end),
                "length": int(end - start + 1),
                "start_time": float(t[start]),
                "end_time": float(t[end]),
            }
        )

    return runs


def select_highest_amplitude_run(
    t: ArrayLike,
    d: ArrayLike,
    runs: list[dict[str, float | int]],
) -> dict[str, Any] | None:
    """Select the strongest run and trim it to start at its largest peak."""
    if not runs:
        return None

    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    def peak_amplitude(run: dict[str, float | int]) -> float:
        start = int(run["start_index"])
        stop = int(run["end_index"]) + 1
        return float(np.max(np.abs(d[start:stop])))

    selected_run = max(runs, key=peak_amplitude)
    run_start = int(selected_run["start_index"])
    run_stop = int(selected_run["end_index"]) + 1
    local_peak_index = int(np.argmax(np.abs(d[run_start:run_stop])))
    peak_index = run_start + local_peak_index

    selected_t_absolute = t[peak_index:run_stop].copy()
    selected_d = d[peak_index:run_stop].copy()
    if selected_t_absolute.size < 2:
        return None

    selected_t = selected_t_absolute - selected_t_absolute[0]
    return {
        "t": selected_t,
        "d": selected_d,
        "peak_amplitude": float(np.abs(d[peak_index])),
        "peak_index": int(peak_index),
        "run_start_index": run_start,
        "run_end_index": int(selected_run["end_index"]),
        "run_start_time": float(selected_run["start_time"]),
        "run_end_time": float(selected_run["end_time"]),
        "original_start_time": float(selected_t_absolute[0]),
        "original_end_time": float(selected_t_absolute[-1]),
    }


def choose_common_analysis_interval(
    windows: list[SubbandWindow],
    t: ArrayLike,
) -> AnalysisInterval | None:
    """Earliest accepted peak through the longest useful duration, covering all peaks."""
    accepted = [window for window in windows if window.accepted]
    if not accepted:
        return None

    t = np.asarray(t, dtype=float).ravel()
    starts = [float(window.trimmed_start_time) for window in accepted]
    ends = [float(window.trimmed_end_time) for window in accepted]
    durations = [
        float(window.trimmed_end_time) - float(window.trimmed_start_time)
        for window in accepted
    ]
    common_start = min(starts)
    longest = max(durations)
    common_end = max(common_start + longest, max(ends))
    common_start = max(common_start, float(t[0]))
    common_end = min(common_end, float(t[-1]))
    if common_end <= common_start:
        return None
    return AnalysisInterval(
        start_time=common_start,
        end_time=common_end,
        duration=common_end - common_start,
        strategy="earliest_peak_plus_longest_useful",
        n_accepted_subbands=len(accepted),
    )


def slice_to_analysis_interval(
    t: ArrayLike,
    d: ArrayLike,
    interval: AnalysisInterval | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict ``t`` and ``d`` to the common analysis window when it exists."""
    t_arr = np.asarray(t, dtype=float).ravel()
    d_arr = np.asarray(d, dtype=float).ravel()
    if interval is None:
        return t_arr, d_arr
    mask = (t_arr >= float(interval.start_time)) & (t_arr <= float(interval.end_time))
    if int(np.count_nonzero(mask)) < 2:
        return t_arr, d_arr
    return t_arr[mask], d_arr[mask]


def detect_subband_windows(
    t: ArrayLike,
    d: ArrayLike,
    min_f: float,
    max_f: float,
    initial_width: float,
    scaling: float,
    min_points: int = 30,
    n_sigma: float = 3.0,
    filter_order: int = 4,
    fft_points: int = 2000,
) -> InitialConditionStage:
    """Filter each dynamic subband and detect its peak-trimmed useful run."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    specs = build_dynamic_subbands(min_f, max_f, initial_width, scaling)

    dt = float(np.median(np.diff(t)))
    nyquist = 0.5 / dt
    if max_f >= nyquist:
        raise ValueError(
            f"max_f ({max_f}) must be below Nyquist ({nyquist})"
        )

    windows: list[SubbandWindow] = []
    for spec in specs:
        d_band = bandpass_subband(
            t,
            d,
            spec.min_f,
            spec.max_f,
            order=filter_order,
        )
        noise_floor, three_sigma, method = get_noise_floor(
            t,
            d_band,
            spec.min_f,
            spec.max_f,
            fft_points=fft_points,
        )
        runs = find_runs(
            t,
            d_band,
            noise_floor=noise_floor,
            n_points=min_points,
            n_sigma=n_sigma,
        )
        selected = select_highest_amplitude_run(t, d_band, runs)
        if selected is None:
            windows.append(
                SubbandWindow(
                    spec=spec,
                    accepted=False,
                    noise_floor=noise_floor,
                    threshold=three_sigma,
                    noise_method=method,
                    filter_order=filter_order,
                    reason="no_accepted_run",
                )
            )
            continue

        windows.append(
            SubbandWindow(
                spec=spec,
                accepted=True,
                noise_floor=noise_floor,
                threshold=three_sigma,
                noise_method=method,
                run_start_index=selected["run_start_index"],
                run_end_index=selected["run_end_index"],
                run_start_time=selected["run_start_time"],
                run_end_time=selected["run_end_time"],
                peak_index=selected["peak_index"],
                peak_time=selected["original_start_time"],
                peak_amplitude=selected["peak_amplitude"],
                trimmed_start_time=selected["original_start_time"],
                trimmed_end_time=selected["original_end_time"],
                n_points=int(selected["t"].size),
                t_local=selected["t"],
                d_local=selected["d"],
                reason="peak_trimmed_run",
                filter_order=filter_order,
            )
        )

    interval = choose_common_analysis_interval(windows, t)
    return InitialConditionStage(
        specs=specs,
        windows=windows,
        interval=interval,
        extras={
            "min_f": float(min_f),
            "max_f": float(max_f),
            "initial_subband_width": float(initial_width),
            "initial_subband_scaling": float(scaling),
            "min_points": int(min_points),
            "n_sigma": float(n_sigma),
            "filter_order": int(filter_order),
            "fft_points": int(fft_points),
            "rcrpy_available": rcrpy is not None,
        },
    )


def save_initial_condition_stage(
    output_dir: Path,
    stage: InitialConditionStage,
) -> None:
    """Write subband edges, time windows, and the common analysis interval."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "subbands.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["subband_index", "min_frequency_hz", "max_frequency_hz", "width_hz"]
        )
        for spec in stage.specs:
            writer.writerow([spec.index, spec.min_f, spec.max_f, spec.width])

    with (output_dir / "time_windows.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "subband_index",
                "min_frequency_hz",
                "max_frequency_hz",
                "accepted",
                "noise_floor",
                "threshold",
                "noise_method",
                "run_start_time",
                "run_end_time",
                "peak_time",
                "peak_amplitude",
                "trimmed_start_time",
                "trimmed_end_time",
                "n_points",
                "reason",
            ]
        )
        for window in stage.windows:
            writer.writerow(
                [
                    window.spec.index,
                    window.spec.min_f,
                    window.spec.max_f,
                    int(window.accepted),
                    window.noise_floor,
                    window.threshold,
                    window.noise_method,
                    window.run_start_time,
                    window.run_end_time,
                    window.peak_time,
                    window.peak_amplitude,
                    window.trimmed_start_time,
                    window.trimmed_end_time,
                    window.n_points,
                    window.reason,
                ]
            )

    if stage.interval is not None:
        (output_dir / "analysis_interval.json").write_text(
            json.dumps(asdict(stage.interval), indent=2) + "\n"
        )

    (output_dir / "stage_config.json").write_text(
        json.dumps(stage.extras, indent=2, sort_keys=True) + "\n"
    )

    for window in stage.windows:
        band_dir = output_dir / f"subband_{window.spec.index:03d}"
        band_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "subband_index": window.spec.index,
            "min_frequency_hz": window.spec.min_f,
            "max_frequency_hz": window.spec.max_f,
            "accepted": window.accepted,
            "reason": window.reason,
            "noise_floor": window.noise_floor,
            "threshold": window.threshold,
            "noise_method": window.noise_method,
            "run_start_time": window.run_start_time,
            "run_end_time": window.run_end_time,
            "peak_time": window.peak_time,
            "peak_amplitude": window.peak_amplitude,
            "trimmed_start_time": window.trimmed_start_time,
            "trimmed_end_time": window.trimmed_end_time,
            "n_points": window.n_points,
            "filter_family": window.filter_family,
            "filter_order": window.filter_order,
            "filter_zero_phase": window.filter_zero_phase,
        }
        (band_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        if (
            window.accepted
            and window.t_local is not None
            and window.d_local is not None
        ):
            np.savetxt(
                band_dir / "selected_timeseries.csv",
                np.column_stack((window.t_local, window.d_local)),
                delimiter=",",
                header="t_local_s,amplitude",
                comments="",
            )
