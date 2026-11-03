# Standard library
import random
from dataclasses import dataclass

# Third-party libraries
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.filter import remez_fir

# Local repository modules
import utils


def get_instrument_response(
    inventory,
    trace,
    f_min,
    f_max,
    f_points,
    output="VEL",
):
    """
    Evaluate the instrument response over a frequency grid.

    Returns
    -------
    f_space : numpy.ndarray
        Frequencies in Hz.
    ir : numpy.ndarray
        Complex instrument response at each frequency.
    """

    if f_points < 2:
        raise ValueError("f_points must be at least 2")

    if f_min <= 0 or f_max <= f_min:
        raise ValueError(
            f"Require 0 < f_min < f_max; got {f_min}, {f_max}"
        )

    # Linear frequency spacing, as requested.
    f_space = np.linspace(f_min, f_max, f_points)

    response = inventory.get_response(
        trace.id,
        trace.stats.starttime,
    )

    ir = response.get_evalresp_response_for_frequencies(
        f_space,
        output=output,
    )

    return f_space, np.asarray(ir)


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

    start_time = UTCDateTime(start_time)
    end_time = UTCDateTime(end_time)

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

    f_space, ir = get_instrument_response(
        inventory=inventory,
        trace=trace,
        f_min=min_f,
        f_max=max_f,
        f_points=1_000,
        output="VEL",
    )

    ir /= max(ir)

    plt.plot(f_space, abs(ir))
    plt.show()

    trace.detrend(
        "spline",
        order=3,
        dspline=1728
    )
    
    # trace.remove_response(inventory=inventory, output="ACC")

    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)

    # trace.filter(
    #     "lowpass_fir",
    #     freqmin=min_f,
    #     freqmax=max_f,
    #     df=trace.stats.sampling_rate
    # )

    delta = float(trace.stats.delta)
    sample_count = len(trace)

    t = np.arange(sample_count, dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)

    # print(trace.stats.sampling_rate)
    # print("sampling rate:", trace.stats.sampling_rate)
    # print("Nyquist:", trace.stats.sampling_rate / 2)
    # print("min_f:", min_f)
    # print("max_f:", max_f)
    # d = remez_fir(d, freqmin=min_f, freqmax=max_f, df=trace.stats.sampling_rate)

    # t = t[:len(d)]
    # print(len(t), len(d))

    return t, d, f_space, ir


def get_best_signal(probability_surface):
    f_space, k_space, log_prob_space = probability_surface

    best_index = jnp.unravel_index(
        jnp.argmax(log_prob_space),
        log_prob_space.shape,
    )

    f_index, k_index = best_index

    return f_space[f_index], k_space[k_index]


def get_log_prob(
    t,
    d,
    signal_space,
    signal,
    f_space,
    ir,
):
    """
    Calculate the log probability for one frequency/decay-rate candidate.
    """

    fs, ks = utils.unpack_signals(signal)

    omegas = fs * 2.0 * jnp.pi

    r = len(signal) * 2
    m = len(signal) * 2

    N = (
        d.shape[0]
        * 2.0
        * (signal_space.f_max - signal_space.f_min)
        * (t[1] - t[0])
    )

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Instrument-response amplitude at the candidate frequency.
    response_amplitude = jnp.abs(ir)

    ir_at_frequency = jnp.interp(
        fs,
        f_space,
        response_amplitude,
    )

    # Prevent division or numerical problems if the response is zero.
    ir_at_frequency = jnp.maximum(
        ir_at_frequency,
        jnp.finfo(d.dtype).tiny,
    )

    # The measured signal is the physical signal multiplied by
    # the instrument-response magnitude.
    G = jnp.vstack((
        jnp.cos(arg) * decay,
        jnp.sin(arg) * decay,
    ))

    g = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(g)

    # Optional numerical protection against very small eigenvalues.
    eigenvalues = jnp.maximum(
        eigenvalues,
        jnp.finfo(eigenvalues.dtype).eps,
    )

    H = (
        eigenvectors / jnp.sqrt(eigenvalues)
    ).T @ G

    h = H @ d

    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    ratio = sum_sq_proj / sum_sq_data

    # Keep ratio inside the valid logarithm range.
    ratio = jnp.clip(ratio, 0.0, 1.0 - jnp.finfo(ratio.dtype).eps)

    return (1 / ir_at_frequency) * 0.5 * (m - N) * jnp.log1p(-ratio)


def grid_search(
        t, 
        d, 
        signal_space,
        f_space, 
        ir,
        f_points, 
        k_points,  
        return_probability_surface=False, 
        batch_size=256
    ):
    '''
    Procedure for identifying candidate signal with a grid search.
    1) Create grid across chosen parameter set of frequencies and decay rates.
    2) Evaluate probability across the grid.
    3) Return signal with highest probability.
    '''

    f_min = signal_space.f_min
    f_max = signal_space.f_max
    k_min = signal_space.k_min
    k_max = signal_space.k_max

    ir = jnp.asarray(ir)

    k_space = jnp.geomspace(k_min, k_max, k_points)

    log_k_space = jnp.linspace(
        jnp.log(k_min),
        jnp.log(k_max),
        k_points,
    )

    f_delta = f_space[1] - f_space[0]
    log_k_delta = log_k_space[1] - log_k_space[0]

    f_grid, log_k_grid = jnp.meshgrid(
        f_space,
        log_k_space,
        indexing="ij",
    )

    rng_key_value = random.randint(1, 10_000)
    key = jax.random.key(random.randint(1, 10_000))
    f_key, k_key = jax.random.split(key)

    f_jitter = jax.random.uniform(
        f_key,
        shape=(f_points, k_points),
        minval=-f_delta / 2,
        maxval=f_delta / 2,
    )

    f_grid = jnp.clip(
        f_grid + f_jitter, 
        f_min, 
        f_max
    )

    log_k_jitter = jax.random.uniform(
        k_key,
        shape=(f_points, k_points),
        minval=-log_k_delta / 2,
        maxval=log_k_delta / 2,
    )

    log_k_grid = jnp.clip(
        log_k_grid + log_k_jitter,
        jnp.log(k_min),
        jnp.log(k_max),
    )
    k_grid = jnp.exp(log_k_grid)

    signal_grid = jnp.stack(
        [f_grid.ravel(), k_grid.ravel()],
        axis=-1,
    )

    evaluate_batch = jax.jit(
        jax.vmap(lambda signal: get_log_prob(t, d, signal_space, signal, f_space, ir))
    )

    results = []

    for start in range(0, signal_grid.shape[0], batch_size):
        batch = signal_grid[start:start + batch_size]
        results.append(evaluate_batch(batch))

    log_probs = jnp.concatenate(results)

    log_prob_space = log_probs.reshape(
        f_space.size,
        k_space.size,
    )

    signal = get_best_signal(
        (f_space, k_space, log_prob_space)
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        np.asarray(f_grid).ravel(),
        np.asarray(k_grid).ravel(),
        s=10,
        alpha=0.6,
    )

    plt.xlabel("Frequency")
    plt.ylabel("Decay rate, $k$")
    plt.yscale("log")
    plt.title("Grid-search sample locations")
    # plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Samples.png", dpi=300)

    if return_probability_surface:
            return signal, (f_space, k_space, log_prob_space)

    return signal


@dataclass
class SignalSpace:
    '''
    Defines the set of allowed frequencies and decay rates that bound 
    the whole of signal-decay rate space.
    '''
    f_min: float
    f_max: float
    k_min: float
    k_max: float


def plot_probability_surface(path, probability_surface, title):
    f_space, k_space, log_prob_space = probability_surface
    log_prob_space = log_prob_space.T

    fig, ax = plt.subplots(figsize=(12, 8))

    heatmap = ax.pcolormesh(
        f_space,
        k_space,
        log_prob_space,
        shading="auto",
        cmap="Grays",
    )

    ax.set_yscale("log")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Decay Rate")
    ax.set_title(title)

    colorbar = fig.colorbar(heatmap, ax=ax)
    colorbar.set_label("Log Probability")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    network = "IU"
    station = "ANMO"
    channel = "LHZ"
    location = "00"
    stream_index = 0
    start_time = "2025-07-30T04:24:52"
    end_time = "2025-08-10T17:24:52"

    f_min = 0.00028
    f_max = 0.00180
    k_min = 4.7e-7
    k_max = 2.5e-5

    signal_space = SignalSpace(f_min=f_min, f_max=f_max, k_min=k_min, k_max=k_max)

    t, d, f_space, ir = observed_data(network, station, channel, location, stream_index, start_time, end_time, f_min, f_max)

    plt.plot(t, d, color="black")
    plt.xlim(t[0], t[-1])
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.savefig("Series.png", dpi=300)
    plt.close()

    signal, probability_surface = grid_search(t, d, signal_space, f_space, ir, f_points=1_000, k_points=100, return_probability_surface=True)

    plot_probability_surface("Surface", probability_surface, "Probability Surface")