from bats import get_statistics, BATS
from dracula import Dracula, StatisticsResult

import numpy as np
import jax.numpy as jnp
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

import pandas as pd


def observed_data(network, 
                  station, 
                  channel, 
                  location, 
                  stream_index, 
                  start_time, 
                  end_time,
                  min_f,
                  max_f
                  ):
    
    client = Client('IRIS')

    inventory = client.get_stations(network=network, station=station, location=location, 
                                            channel=channel, starttime=start_time, endtime=end_time, level='response')
    
    stream = client.get_waveforms(network=network, station=station, location=location, 
                                        channel=channel, starttime=start_time, endtime=end_time)
    
    trace = stream[stream_index]
    trace.detrend('constant')
    trace.remove_response(inventory=inventory, output="ACC")
    
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)

    trace.filter('bandpass', freqmin=min_f, freqmax=max_f)

    delta = trace.stats.delta 
    N = len(trace)

    t = jnp.arange(N) * delta
    d = jnp.array(trace.data)

    return t, d


def synthetic_data():

    # TODO: implement synthetic data fetching

    pass



if __name__ == "__main__":
    network = "IU"
    station = "KIP"
    location = "00"
    channel = "LHZ"
    stream_index = 0
    
    # Time frame covering background + event
    start_time = UTCDateTime('2025-07-31T06:24:50')
    end_time = UTCDateTime('2025-08-6T05:24:50')

    min_f=0.000780
    max_f=0.000830

    t, d = observed_data(network, 
                         station, 
                         channel, 
                         location, 
                         stream_index, 
                         start_time, 
                         end_time, 
                         min_f, 
                         max_f)

    bats = BATS(t, d, [0,1], [2,3])
    result = bats.run_grid_search(
        min_f=0.000780,
        max_f=0.000830,
        min_k=1e-5,
        max_k=6e-5,
        f_points=500,
        k_points=500,
        signals=5,
        apply_bandpass=False,
        selection="best",
        diagnostics=True,
    )

    print(result.fs)
    print(result.ks)
    print(result.extras["selected_log_prob"])

    fs = result.fs
    ks = result.ks
    log_probs = result.extras["selected_log_prob"]

    sort_indices = jnp.argsort(log_probs)[::-1]

    # 3. Apply the sorted indices to your arrays
    sorted_fs = fs[sort_indices]
    sorted_ks = ks[sort_indices]
    sorted_log_probs = log_probs[sort_indices]

    model = Dracula(
        t,
        d,
        sorted_fs[::-1],
        sorted_ks[::-1], 
    )
    model.dispatch(
        f_per_worker=5,
        min_signals=1,
        max_signals=len(sorted_fs),
        f_bw=0.000015,
        k_bw=3e-5,
        W=1_000,
        S=2_000,
        sort_signals=False,
        prior_n_std=1,
        unbounded=True
    )