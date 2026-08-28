from bats import statistics, log_prob
from dracula import Dracula

import jax
import jax.numpy as jnp
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

import pandas as pd
import matplotlib.pyplot as plt


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

    stats = statistics(t, d, fs, ks)
    p_spec = stats[3]

    print("Probability: ", stats[0], "\nVariance: ", stats[1], "\nSNR: ", stats[2])

    f, p = p_spec.T

    f = jnp.array(f)
    p = jnp.array(p)

    find_p_idx = jax.vmap(lambda target: jnp.argmin(jnp.abs(f - target)))
    p_idx = find_p_idx(jnp.array(fs))

    ps = p[p_idx]

    sort_order = jnp.argsort(ps)[::-1]
    sort_fs = jnp.array(fs)[sort_order]
    sort_ps = ps[sort_order]
    sort_ks = jnp.array(ks)[sort_order]
    sort_fs_unc = jnp.array(fs_unc)[sort_order]
    sort_ks_unc = jnp.array(ks_unc)[sort_order]

    # Get the indices that sort f in ascending order
    sort_indices = jnp.argsort(f)

    # Apply the sorting indices to both arrays
    f_sorted = f[sort_indices]
    p_sorted = p[sort_indices]

    # Plot the newly sorted data
    plt.plot(f_sorted, p_sorted)
    plt.plot(sort_fs, sort_ps)
    plt.scatter(f_sorted, p_sorted, s=5)
    plt.ylabel("Power")
    plt.xlabel("Frequency (Hz)")
    plt.title("Spectral Power Density")
    plt.show()

    model = Dracula(t, 
                    d, 
                    sort_fs, 
                    sort_ks)
    results = model.dispatch(f_per_worker=5, 
                             min_signals=len(fs) - 5, 
                             max_signals=len(fs), 
                             f_bw=sort_fs_unc, 
                             k_bw=sort_ks_unc, 
                             W=1000,
                             S=2000,
                             max_cores=8)

    print(results)

    fitted_f, fitted_p = results[0][3].T
    print(fitted_f)

    sort_indices = jnp.argsort(fitted_f)
    fitted_f_sorted = fitted_f[sort_indices]
    fitted_p_sorted = fitted_p[sort_indices]

    plt.plot(f_sorted, p_sorted)
    plt.scatter(f_sorted, p_sorted, s=5)

    plt.plot(fitted_f_sorted, fitted_p_sorted)
    plt.scatter(fitted_f_sorted, fitted_p_sorted, s=5)

    plt.ylabel("Power")
    plt.xlabel("Frequency (Hz)")
    plt.title("Spectral Power Density")
    plt.show()