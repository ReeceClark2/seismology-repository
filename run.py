# Internal Python libraries
import shutil
import sys
import tomllib
from pathlib import Path

# External libraries
import numpy as np
from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

# Repository files
from dracula import Dracula, GridSearchArgs, NUTSArgs


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


def main(config_path):
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {config_path}"
        )

    with config_path.open("rb") as file:
        config = tomllib.load(file)

    data_config = config["data"]
    data_config["start_time"] = UTCDateTime(data_config["start_time"])
    data_config["end_time"] = UTCDateTime(data_config["end_time"])

    t, d = observed_data(**data_config)

    model = Dracula(
        t,
        d,
        **config["model"],
    )

    grid_search_args = GridSearchArgs(
        **config["grid_search"],
    )

    nuts_args = {}

    for name in ("sample", "reconcile"):
        nuts_config = config["nuts"][name]

        if "max_tree_depth" in nuts_config["nuts_kwargs"]:
            nuts_config["nuts_kwargs"]["max_tree_depth"] = tuple(
                nuts_config["nuts_kwargs"]["max_tree_depth"]
            )

        nuts_args[name] = NUTSArgs(**nuts_config)

    # Optional init configuration
    init_config = config["nuts"].get("init")

    if init_config is not None:
        if "max_tree_depth" in init_config["nuts_kwargs"]:
            init_config["nuts_kwargs"]["max_tree_depth"] = tuple(
                init_config["nuts_kwargs"]["max_tree_depth"]
            )

        nuts_args["init"] = NUTSArgs(**init_config)
    else:
        nuts_args["init"] = None

    execute_args = config["execute"]
    execute_args["grid_search_args"] = grid_search_args
    execute_args["nuts_args_init"] = nuts_args["init"]
    execute_args["nuts_args_sample"] = nuts_args["sample"]
    execute_args["nuts_args_reconcile"] = nuts_args["reconcile"]

    dracula_output_path = Path(config["model"]["path"])
    parameter_txt_path = dracula_output_path / f"{config_path.stem}.txt"
    shutil.copy2(config_path, parameter_txt_path)

    print(f"Configuration copied to: {parameter_txt_path}")

    model.execute(**execute_args)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python run.py {toml}\n"
            "Example: python run.py dracula_config.toml"
        )

    main(sys.argv[1])
