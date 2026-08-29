# BATS

Bayesian signal processing for time series, built around G. Larry Bretthorst’s methods and sampled with NumPyro NUTS. The main target is low-frequency seismograms of Earth’s normal modes: given data `d` over time `t`, recover frequencies, decay rates, and model diagnostics.

## Layout

| File | Role |
|---|---|
| `bats.py` | Bretthorst likelihood, NumPyro NUTS sampler (`BATS`), and global statistics |
| `colony.py` | Splits an N-signal model into nearby-frequency workers and bandpass-filters the data |
| `dracula.py` | Top-level orchestrator: ranks modes, runs Colonies in parallel, writes outputs |
| `test2.py` | Example: download IRIS data and run `Dracula.dispatch` |
| `test.py` | Standalone mode-by-mode SNR / uncertainty plotting script |
| `environment.yml` | Conda environment |

## How it works

1. **`Dracula`** takes candidate frequencies and decay rates. With `sort_signals=True` (default) it ranks them by spectral power so stronger modes enter first.
2. For each N-signal model in `[min_signals, max_signals]`, a **`Colony`** job is submitted to the process pool. Colony sorts those N modes by frequency (nearby lines are correlated) and chunks them with `f_per_worker`.
3. Each chunk gets a Butterworth bandpass around its band, then a **`BATS`** NUTS run on a small Gram matrix instead of one huge \(2N \times 2N\) matrix.
4. When every BATS worker for that N finishes, `get_statistics` computes log probability, noise variance, SNR, parameter uncertainties, power spectrum, and global likelihood (`glob_LL`).
5. Results are written under `output_dir`.

## Install

```bash
conda env create -f environment.yml
conda activate bats
```

Requires Python 3.11 and the conda-forge stack in `environment.yml` (JAX, NumPyro, SciPy, ObsPy, and plotting/data libraries).

## Quick start

```python
from dracula import Dracula

model = Dracula(t, d, f_init, k_init)
results = model.dispatch(
    f_per_worker=5,
    min_signals=10,
    max_signals=15,
    f_bw=1e-4,          # scalar, or one value per mode
    k_bw=1e-5,
    W=1000,             # NUTS warmup
    S=2000,             # NUTS samples
    prior_n_std=5,      # TruncatedNormal walls at mean ± n_std * scale
    sort_signals=True,
    output_dir="dracula_output",
)

n_stats = results[12]           # N = 12 signal model
print(n_stats.SNR, results.extras["output_dir"])
```

`f_bw` and `k_bw` may be a single scalar (used for every mode) or an array aligned with `f_init` / `k_init`. If omitted, Colony defaults to `1e-3` and `1e-5`.

Frequencies and decay rates are sampled from `TruncatedNormal` priors with hard walls at `mean ± prior_n_std * scale` (default 5). Decay rates are also truncated below at 0.

NumPyro knobs can be passed as `**kwargs` without listing every sampler argument:

```python
results = model.dispatch(
    f_per_worker=3,
    min_signals=8,
    max_signals=12,
    target_accept_prob=0.9,
    nuts_kwargs={"max_tree_depth": 10},
    mcmc_kwargs={"progress_bar": True},
)
```

Time `t` is assumed to be in seconds. Sampling rate for the Butterworth filter is inferred from `median(diff(t))`.

## Outputs

`dispatch` creates `output_dir` (or `./dracula_output` if omitted):

| File | Contents |
|---|---|
| `global_stats.csv` | `N`, `SNR`, `variance`, `glob_LL` |
| `N012_signals.csv` | frequencies, frequency uncertainties, decay rates, decay rate uncertainties |
| `N012_timeseries.png` | data, Bretthorst model \(h \cdot H\), residual; SNR and variance on the plot |
| `N012_power_spectrum.png` | power spectrum before vs after sampling, with original and fitted frequencies |

In-memory results are a `DraculaResult` keyed by signal count: `results[n]`, `results.as_list()`, or `results.extras`.

Each `StatisticsResult` includes `log_prob`, `variance`, `SNR`, `p_spec`, `glob_LL`, `fs`, `ks`, `f_unc`, and `k_unc`.

## Example scripts

- `python test2.py` — fetch an IU.KIP LHZ seismogram from IRIS, load a normal-mode table, and run Dracula. The CSV path `data/earth_normal_modes_table.csv` lives on the `legacy` branch.
- `python test.py` — per-mode bandpass, SNR, and uncertainty ellipses.

## Branches

| Branch | Contents |
|---|---|
| `main` / `dev` | Current Python sources (this layout) |
| `legacy` | Earlier prototype directories, `data/`, and `timeseries-kamchatka/` |
