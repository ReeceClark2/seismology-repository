import multiprocessing

from obspy.core import UTCDateTime

from dracula import Dracula
from initial_conditions import observed_data


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

    model = Dracula(t, d)
    result = model.dispatch(
        signals_per_worker=8,
        min_f=min_f,
        max_f=max_f,
        initial_subband_width=5e-5,
        initial_subband_scaling=4.0,
        signal_count_mode="automatic",
        signal_count_method="global_likelihood",
        selection_fallback=None,
        max_candidates_per_subband=10,
        ic_max_k=1.2e-4,
        ic_f_points=500,
        ic_k_points=500,
        bounds_mode="initial_subband",
        W=1_000,
        S=2_000,
        calc_stats=True,
        stats_at_end=True,
        max_cores=32,
        prior_n_std=1.0,
        imposed_surface="uniform",
        progress_mode="main",
        output_dir="dracula_output",
    )

    n_detected = int(result.extras["f_init"].size)
    selected_n = result.extras.get("selected_n")
    print(f"Detected {n_detected} candidate signals")
    if selected_n is not None:
        stats = result[selected_n]
        print(
            f"Selected N={selected_n}: "
            f"SNR={float(stats.SNR):.6g}, "
            f"glob_LL={float(stats.glob_LL):.6g}"
        )
    print(f"Results saved to: {result.extras['output_dir']}")
    print(f"Final results: {result.extras['final_results_dir']}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
