import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

from dracula import Dracula, NUTSArgs, GridSearchArgs


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
        corners=2,
        zerophase=True,
    )

    delta = float(trace.stats.delta)
    sample_count = len(trace)

    t = np.arange(sample_count, dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)

    return t, d

def main():
    network = "IU"
    station = "KIP"
    location = "00"
    channel = "LHZ"
    stream_index = 0

    start_time = UTCDateTime("2025-07-30T01:24:50")
    end_time = UTCDateTime("2025-08-05T20:24:50")

    f_min = 0.003
    f_max = 0.004

    t, d = observed_data(
        network=network,
        station=station,
        channel=channel,
        location=location,
        stream_index=stream_index,
        start_time=start_time,
        end_time=end_time,
        min_f=f_min,
        max_f=f_max,
    )

    model = Dracula(
        t, 
        d,
        f_min=0.0027,
        f_max=0.0043,
        k_min=1e-5,
        k_max=1e-4,
        max_workers=32
    )

    grid_search_args = GridSearchArgs(
        f_points=50,
        k_points=50
    )
    nuts_kwargs = dict(
        target_accept_prob=0.75,
    )
    mcmc_kwargs = dict(
        num_warmup=20,
        num_samples=60,
        num_chains=1,
        chain_method="parallel"
    )
    run_kwargs = dict()

    nuts_args_init = NUTSArgs(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        run_kwargs=run_kwargs,
        seed=5
    )

    nuts_kwargs = dict(
        target_accept_prob=0.85,
    )
    mcmc_kwargs = dict(
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        chain_method="parallel"
    )
    run_kwargs = dict()

    nuts_args_sample = NUTSArgs(
        nuts_kwargs=nuts_kwargs,
        mcmc_kwargs=mcmc_kwargs,
        run_kwargs=run_kwargs,
        seed=18
    )

    model.execute(
        subband_count=10, 
        subband_scaling_factor=1.0,
        depth=8,
        grid_search_args=grid_search_args,
        cores_per_initial_conditions_worker=1,

        signals_per_block=16,
        fill_order=0,
        nuts_args_sample=nuts_args_sample,
        cores_per_sample_worker=1,

        perform_minimize=False,
    )


if __name__ == "__main__":
    main()
