from bats import get_statistics
from dracula import Dracula

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
    end_time = UTCDateTime('2025-08-3T05:24:50')

    min_f = 0.003
    max_f = 0.004

    t, d = observed_data(network, 
                         station, 
                         channel, 
                         location, 
                         stream_index, 
                         start_time, 
                         end_time, 
                         min_f, 
                         max_f)

    print("Data collected!")

    df = pd.read_csv("data/earth_normal_modes_table.csv")

    condition = (
        (df["f_obs"] > 1e6 * min_f)
        & (df["f_obs"] < 1e6 * max_f)
        & df["f_obs"].notna()
        & df["k_obs"].notna()
    )

    df = df[condition]

    fs = df['f_obs'].values / 1e6
    ks = df['k_obs'].values
    fs_unc = df['f_unc'].values / 1e6
    ks_unc = df['k_unc'].values

    print("Statistics started...")

    stats = get_statistics(t, d, fs, ks)
    print("Probability: ", stats.log_prob, "\nVariance: ", stats.variance, "\nSNR: ", stats.SNR)

    model = Dracula(t, d, fs, ks)
    results = model.dispatch(
        f_per_worker=5,
        min_signals=len(fs) - 5,
        max_signals=len(fs),
        f_bw=fs_unc,
        k_bw=ks_unc,
        W=1000,
        S=2000,
        max_cores=8,
        sort_signals=True,
        output_dir="dracula_output",
    )

    print("Wrote outputs to", results.extras["output_dir"])