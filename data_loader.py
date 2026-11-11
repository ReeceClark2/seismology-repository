import argparse
from pathlib import Path

import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime


EVENT = "kamchatka"
CHANNEL = "LHZ"
LOCATION = "00"
STREAM_INDEX = 0
START_TIME = "2025-07-30T04:24:52"
END_TIME = "2025-08-14T04:24:52"


def get_observed_data(network, station):
    start_time = UTCDateTime(START_TIME)
    end_time = UTCDateTime(END_TIME)
    client = Client("EarthScope")

    inventory = client.get_stations(
        network=network,
        station=station,
        location=LOCATION,
        channel=CHANNEL,
        starttime=start_time,
        endtime=end_time,
        level="response",
    )

    stream = client.get_waveforms(
        network=network,
        station=station,
        location=LOCATION,
        channel=CHANNEL,
        starttime=start_time,
        endtime=end_time,
    )

    trace = stream[STREAM_INDEX]
    trace.detrend("spline", order=3, dspline=1728)
    trace.remove_response(inventory=inventory, output="ACC")

    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    delta = float(trace.stats.delta)
    t = np.arange(len(trace), dtype=float) * delta
    d = np.asarray(trace.data, dtype=float)

    output_dir = Path.cwd() / "observed" / EVENT
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{network}_{station}_TS.ascii"
    np.savetxt(output_path, np.column_stack((t, d)))

    print(f"Saved {len(t)} samples to {output_path}")
    return t, d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and save observed waveform data."
    )
    parser.add_argument("network")
    parser.add_argument("station")
    args = parser.parse_args()

    get_observed_data(args.network, args.station)