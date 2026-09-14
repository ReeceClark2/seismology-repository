import csv
import multiprocessing
from pathlib import Path

import numpy as np
from obspy.core import UTCDateTime
import jax

from bats import get_statistics
from dracula import Dracula
from initial_conditions import observed_data


def read_initial_conditions(
    csv_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Read frequency and decay-rate initial conditions from the grid CSV."""
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = []

    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)

        required_columns = {
            "subband_index",
            "signal_index",
            "frequency_hz",
            "decay_rate",
        }

        missing = required_columns.difference(reader.fieldnames or [])

        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )

        for line_number, row in enumerate(reader, start=2):
            try:
                subband_index = int(row["subband_index"])
                signal_index = int(row["signal_index"])
                frequency = float(row["frequency_hz"])
                decay_rate = float(row["decay_rate"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid value on CSV line {line_number}"
                ) from error

            if not np.isfinite(frequency):
                raise ValueError(
                    f"Non-finite frequency on CSV line {line_number}"
                )

            if not np.isfinite(decay_rate):
                raise ValueError(
                    f"Non-finite decay rate on CSV line {line_number}"
                )

            if frequency <= 0.0:
                raise ValueError(
                    f"Frequency must be positive on line {line_number}"
                )

            if decay_rate < 0.0:
                raise ValueError(
                    f"Decay rate cannot be negative on line {line_number}"
                )

            rows.append(
                (
                    subband_index,
                    signal_index,
                    frequency,
                    decay_rate,
                )
            )

    if not rows:
        raise ValueError(f"No initial conditions found in {csv_path}")

    # Preserve deterministic subband/signal ordering.
    rows.sort(key=lambda item: (item[0], item[1]))

    frequencies = np.asarray(
        [row[2] for row in rows],
        dtype=float,
    )
    decay_rates = np.asarray(
        [row[3] for row in rows],
        dtype=float,
    )

    return frequencies, decay_rates


def main():
    # Use the same observed-data parameters as initial_conditions.py.
    network = "IU"
    station = "KIP"
    location = "00"
    channel = "LHZ"
    stream_index = 0

    start_time = UTCDateTime("2025-07-29T23:24:50")
    end_time = UTCDateTime("2025-08-06T05:24:50")

    min_f = 0.0030
    max_f = 0.0040

    csv_path = Path("data/all_subband_grid_results.csv")

    print("Downloading and preparing observed data...")

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

    frequencies, decay_rates = read_initial_conditions(csv_path)

    print(
        f"Testing get_statistics for {frequencies.size} signals "
        f"({2 * frequencies.size} parameters)..."
    )

    initial_stats = get_statistics(
        t,
        d,
        frequencies,
        decay_rates,
    )

    # Force all asynchronous JAX calculations to finish now.
    for value in (
        initial_stats.log_prob,
        initial_stats.variance,
        initial_stats.SNR,
        initial_stats.p_spec,
        initial_stats.glob_LL,
        initial_stats.cov_mat,
        initial_stats.f_unc,
        initial_stats.k_unc,
    ):
        jax.block_until_ready(value)

    print(
        "Initial statistics completed successfully: "
        f"SNR={float(initial_stats.SNR):.6g}, "
        f"log_prob={float(initial_stats.log_prob):.6g}, "
        f"glob_LL={float(initial_stats.glob_LL):.6g}"
    )

    del initial_stats

    print(f"Loaded {frequencies.size} initial conditions")
    print(
        f"Frequency range: "
        f"{frequencies.min():.8g}–{frequencies.max():.8g} Hz"
    )
    print(
        f"Decay-rate range: "
        f"{decay_rates.min():.8g}–{decay_rates.max():.8g}"
    )

    model = Dracula(
        t=t,
        d=d,
        f_init=frequencies,
        k_init=decay_rates,
    )

    # Adjust these arguments as needed.
    result = model.dispatch(
        f_per_worker=8,
        min_signals=1,
        max_signals=50,
        f_bw=2.5e-5,
        k_bw=2,
        W=1_000,
        S=2_000,
        calc_stats=True,
        stats_at_end=True,
        max_cores=32,

        sort_signals=False,

        output_dir="dracula_output",
        prior_n_std=1.0,
        unbounded=True,
    )

    print(f"Results saved to: {result.extras['output_dir']}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
