from dataclasses import dataclass

import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

import csv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from bats import BATS, DECAY_FLOOR
from subbands import (
    bandpass_subband,
    find_runs,
    get_noise_floor,
    select_highest_amplitude_run,
)


@dataclass
class SubbandGridResult:
    band_index: int
    min_f: float
    max_f: float
    fs: np.ndarray
    ks: np.ndarray
    log_probs: np.ndarray
    extras: dict


def estimate_min_decay_rate(
    t,
    d,
    threshold,
) -> float:
    """Estimate k from A(T) = A(0) exp(-kT), floored above zero."""
    t = np.asarray(t, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()

    duration = float(t[-1] - t[0])
    peak = float(np.max(np.abs(d)))

    if duration <= 0.0:
        raise ValueError("Subband duration must be positive")

    if peak <= 0.0 or threshold <= 0.0:
        return float(DECAY_FLOOR)

    # k = log(A_initial / A_final) / duration.
    # If peak <= threshold, there is no positive lower bound on k.
    return float(max(DECAY_FLOOR, np.log(peak / threshold) / duration))


def save_subband_grid_results(
    path,
    results,
):
    """Save all selected frequencies, decay rates, and log probabilities."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)

        writer.writerow(
            [
                "subband_index",
                "subband_min_frequency_hz",
                "subband_max_frequency_hz",
                "signal_index",
                "frequency_hz",
                "decay_rate",
                "log_probability",
            ]
        )

        for result in sorted(results, key=lambda item: item.band_index):
            if not (
                result.fs.size
                == result.ks.size
                == result.log_probs.size
            ):
                raise ValueError(
                    f"Subband {result.band_index} returned mismatched "
                    "fs, ks, and log-probability lengths"
                )

            for signal_index, (frequency, decay_rate, log_prob) in enumerate(
                zip(result.fs, result.ks, result.log_probs),
                start=1,
            ):
                writer.writerow(
                    [
                        result.band_index,
                        result.min_f,
                        result.max_f,
                        signal_index,
                        float(frequency),
                        float(decay_rate),
                        float(log_prob),
                    ]
                )


def run_subband_grid_search(job):
    """Run one JAX-accelerated BATS grid search in a worker process."""
    band_index, subband, grid_options = job

    subband_t = np.asarray(subband.t, dtype=float)
    subband_d = np.asarray(subband.d, dtype=float)

    min_k = estimate_min_decay_rate(
        subband_t,
        subband_d,
        subband.threshold,
    )

    max_k = float(grid_options["max_k"])

    # Ensure the grid has a nonzero interval.
    min_k = min(min_k, max_k * 0.999)

    model = BATS(
        subband_t,
        subband_d,
        [0.0],
        [0.0],
    )

    result = model.run_grid_search(
        min_f=float(subband.min_f),
        max_f=float(subband.max_f),
        min_k=min_k,
        max_k=max_k,
        f_points=int(grid_options["f_points"]),
        k_points=int(grid_options["k_points"]),
        signals=int(grid_options["signals"]),
        apply_bandpass=False,
        grid_batch_size=int(grid_options["grid_batch_size"]),
        diagnostics=bool(grid_options["diagnostics"]),
        progress_bar=bool(grid_options["progress_bar"]),
        progress_position=None,
    )

    log_probs = np.asarray(
        result.extras["selected_log_prob"],
        dtype=float,
    ).ravel()

    return SubbandGridResult(
        band_index=band_index,
        min_f=float(subband.min_f),
        max_f=float(subband.max_f),
        fs=np.asarray(result.fs, dtype=float).ravel(),
        ks=np.asarray(result.ks, dtype=float).ravel(),
        log_probs=log_probs,
        extras=result.extras,
    )


@dataclass
class SelectedSubband:
    min_f: float
    max_f: float
    t: np.ndarray
    d: np.ndarray
    noise_floor: float
    threshold: float
    peak_amplitude: float
    original_start_time: float
    original_end_time: float


def observed_data(
    network,
    station,
    channel,
    location,
    stream_index,
    start_time,
    end_time,
    min_f,
    max_f,
):
    """Download, response-correct, decimate, and broadly bandpass the data."""
    client = Client("IRIS")

    inventory = client.get_stations(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=start_time,
        endtime=end_time,
        level="response",
    )

    stream = client.get_waveforms(
        network=network,
        station=station,
        location=location,
        channel=channel,
        starttime=start_time,
        endtime=end_time,
    )

    trace = stream[stream_index]
    trace.detrend("constant")
    trace.remove_response(inventory=inventory, output="ACC")

    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)

    trace.filter(
        "bandpass",
        freqmin=min_f,
        freqmax=max_f,
        corners=16,
        zerophase=True,
    )

    delta = float(trace.stats.delta)
    sample_count = len(trace)

    t = np.arange(sample_count, dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)

    return t, d


def extract_selected_subbands(
    t,
    d,
    min_f,
    max_f,
    n_subbands=10,
    min_points=20,
    n_sigma=3.0,
    filter_order=4,
    fft_points=2000,
):
    """Extract the highest-amplitude accepted run from every subband."""
    subband_edges = np.linspace(
        min_f,
        max_f,
        n_subbands + 1,
    )

    selected_subbands: list[SelectedSubband] = []

    for band_index, (subband_min, subband_max) in enumerate(
        zip(subband_edges[:-1], subband_edges[1:]),
        start=1,
    ):
        d_subband = bandpass_subband(
            t,
            d,
            min_f=subband_min,
            max_f=subband_max,
            order=filter_order,
        )

        noise_floor, three_sigma, _method = get_noise_floor(
            t,
            d_subband,
            min_f=subband_min,
            max_f=subband_max,
            fft_points=fft_points,
        )

        runs = find_runs(
            t,
            d_subband,
            noise_floor=noise_floor,
            n_points=min_points,
            n_sigma=n_sigma,
        )

        selected = select_highest_amplitude_run(
            t,
            d_subband,
            runs,
        )

        if selected is None:
            print(
                f"Subband {band_index:02d}, "
                f"{subband_min * 1000:.4f}–"
                f"{subband_max * 1000:.4f} mHz: "
                "no accepted run"
            )
            continue

        selected_subbands.append(
            SelectedSubband(
                min_f=float(subband_min),
                max_f=float(subband_max),
                t=selected["t"],
                d=selected["d"],
                noise_floor=noise_floor,
                threshold=n_sigma * noise_floor,
                peak_amplitude=selected["peak_amplitude"],
                original_start_time=selected["original_start_time"],
                original_end_time=selected["original_end_time"],
            )
        )

        print(
            f"Subband {band_index:02d}, "
            f"{subband_min * 1000:.4f}–"
            f"{subband_max * 1000:.4f} mHz: "
            f"{selected['t'].size} points, "
            f"peak={selected['peak_amplitude']:.6g}"
        )

    return selected_subbands


def main():
    network = "IU"
    station = "KIP"
    location = "00"
    channel = "LHZ"
    stream_index = 0

    start_time = UTCDateTime("2025-07-29T23:24:50")
    end_time = UTCDateTime("2025-08-06T05:24:50")

    min_f = 0.0030
    max_f = 0.0040

    t, d = observed_data(
        network=network,
        station=station,
        channel=channel,
        location=location,
        stream_index=stream_index,
        start_time=start_time,
        end_time=end_time,
        min_f=min_f,
        max_f=max_f,
    )

    selected_subbands = extract_selected_subbands(
        t=t,
        d=d,
        min_f=min_f,
        max_f=max_f,
        n_subbands=20,
        min_points=30,
        n_sigma=3.0,
        filter_order=4,
        fft_points=2000,
    )

    n_workers = 6

    grid_options = {
        "max_k": 1.2e-4,
        "f_points": 500,
        "k_points": 500,
        "signals": 5,
        "grid_batch_size": 4_096,
        "diagnostics": True,

        # Disable nested progress bars because multiple subprocess bars
        # generally overwrite each other.
        "progress_bar": True,
    }

    jobs = [
        (band_index, subband, grid_options)
        for band_index, subband in enumerate(
            selected_subbands,
            start=1,
        )
    ]

    subband_results = []

    # Windows requires the spawn context for JAX and safe process creation.
    context = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=context,
    ) as executor:
        future_to_job = {
            executor.submit(run_subband_grid_search, job): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            band_index, subband, _ = future_to_job[future]

            try:
                result = future.result()
            except Exception as error:
                print(
                    f"Subband {band_index}, "
                    f"{subband.min_f * 1000:.4f}–"
                    f"{subband.max_f * 1000:.4f} mHz failed: {error}"
                )
                continue

            subband_results.append(result)

            print(
                f"Finished subband {result.band_index}: "
                f"{result.min_f * 1000:.4f}–"
                f"{result.max_f * 1000:.4f} mHz, "
                f"{result.fs.size} signals"
            )

    save_subband_grid_results(
        "all_subband_grid_results.csv",
        subband_results,
    )

    print("Saved results to all_subband_grid_results.csv")


if __name__ == "__main__":
    main()
