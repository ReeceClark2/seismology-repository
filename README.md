# BATS

Bayesian signal processing for time series, built around G. Larry Bretthorst’s methods and sampled with NumPyro NUTS. The main target is low-frequency seismograms of Earth’s normal modes: given data `d` over time `t`, recover frequencies, decay rates, and model diagnostics.

## Layout

| File | Role |
|---|---|
| `bats.py` | Bretthorst likelihood, NumPyro NUTS sampler (`BATS`), and global statistics |
| `colony.py` | Splits an N-signal model into nearby-frequency workers and bandpass-filters the data |
| `dracula.py` | Top-level orchestrator: ranks modes, runs Colonies in parallel, writes outputs |
| `subbands.py` | Dynamic initial-condition subbands and useful-time windows |
| `detection.py` | Serial `(f, ln k)` grid search on subband residuals |
| `run.py` | Example: download IRIS data and run `Dracula.dispatch` |
| `environment.yml` | Conda environment |

## How it works

1. **`Dracula`** takes candidate frequencies and decay rates. With `sort_signals=True` (default) it ranks them by spectral power so stronger modes enter first.
2. For each N-signal model in `[min_signals, max_signals]`, a **`Colony`** job is submitted to the process pool. Colony sorts those N modes by frequency (nearby lines are correlated) and chunks them with `signals_per_worker`.
3. Each chunk gets a Butterworth bandpass around its band, then a **`BATS`** NUTS run on a small Gram matrix instead of one huge \(2N \times 2N\) matrix.
4. When every BATS worker for that N finishes, `get_statistics` computes log probability, noise variance, SNR, parameter uncertainties, power spectrum, and global likelihood (`glob_LL`).
5. Results are written under a timestamped output tree created at the start of `dispatch`.

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
    signals_per_worker=5,
    min_signals=10,
    max_signals=15,
    f_bw=1e-4,          # scalar, or one value per mode
    k_bw=1e-5,
    W=1000,             # NUTS warmup
    S=2000,             # NUTS samples
    prior_n_std=5,      # TruncatedNormal walls at mean ± n_std * scale
    sort_signals=True,
    progress_mode="main",
    output_dir="dracula_output",
)

n_stats = results[12]           # N = 12 signal model
print(n_stats.SNR, results.extras["output_dir"])
```

`Dracula(t, d)` is valid; both `f_init` and `k_init` must be supplied together, or both omitted. Without initial conditions, `dispatch` runs Stage C (windows), Stage D (grid-search candidates), then Stage E (Colony/BATS). If you omit `signals` / `min_signals`/`max_signals`, the detected count is used.

```python
model = Dracula(t, d)
result = model.dispatch(
    signals_per_worker=5,
    min_f=0.0002,
    max_f=0.0100,
    initial_subband_width=0.0001,
    initial_subband_scaling=4.0,
    signal_count_mode="automatic",
    signal_count_method="peak_contrast",
    bounds_mode="initial_subband",
    n_chains=2,
    progress_mode="main",
)
print(result.extras["selected_n"], result.extras["final_results_dir"])
```

Use `signals=32` for one model size, or `min_signals` and `max_signals` for a range. Those two styles cannot be combined.

`f_bw` and `k_bw` may be a single scalar (used for every mode) or an array aligned with `f_init` / `k_init`. If omitted, Colony defaults to `1e-3` and `1e-5`. **`k_bw` is a natural-log decay bandwidth**, not a linear \(k\) width. `f_per_worker` is a deprecated alias of `signals_per_worker`.

NUTS samples frequency linearly and decay as `log_ks`; the physical rate is `k = exp(log_k)`. Priors are bounded `imposed_surface="gaussian"` (truncated normal) or `"uniform"`. `unbounded=True/False` is a deprecated alias for those two surfaces. Decay uncertainty is reported as `\sigma_{\ln k}` and the multiplicative interval \(k/e^{\sigma}\) to \(k \times e^{\sigma}\). Linear `k_unc` remains as a first-order compatibility field \(k\,\sigma_{\ln k}\). Covariance uses parameter order `[f_1…f_r, log_k_1…log_k_r]`.

NumPyro knobs can be passed as `**kwargs` without listing every sampler argument:

```python
results = model.dispatch(
    signals_per_worker=3,
    min_signals=8,
    max_signals=12,
    target_accept_prob=0.9,
    nuts_kwargs={"max_tree_depth": 10},
    mcmc_kwargs={"progress_bar": True},
)
```

Time `t` is assumed to be in seconds. Sampling rate for the Butterworth filter is inferred from `median(diff(t))`.

## Outputs

`dispatch` creates a timestamped directory from `output_dir` (or `./dracula_output` if omitted) **before** sampling:

| Path | Contents |
|---|---|
| `run_configuration/run_config.json` | Validated dispatch settings |
| `initial_conditions/` | Stage C windows plus Stage D `selected_candidates.csv` and per-signal PNG diagnostics |
| `sampling/global_stats.csv` | `N`, `SNR`, `variance`, `glob_LL` |
| `sampling/N012_signals.csv` | frequencies, frequency and log-decay uncertainties, multiplicative decay interval |
| `sampling/N012_timeseries.png` | data, Bretthorst model \(h \cdot H\), residual; SNR and variance on the plot |
| `sampling/N012_power_spectrum.png` | power spectrum before vs after sampling, with original and fitted frequencies |
| `sampling/chain_diagnostics.csv` | Independent NUTS chains: seed, min potential energy, finite-sample count |
| `final_results/` | Best-N model: signals CSV, residual, covariance, timeseries/spectrum/f–k plots |

`progress_mode` is `"none"`, `"main"` (Dracula bar only), or `"detailed"` (worker/NUTS bars too).

In-memory results are a `DraculaResult` keyed by signal count: `results[n]`, `results.as_list()`, or `results.extras`.

Each `StatisticsResult` includes `log_prob`, `variance`, `SNR`, `p_spec`, `glob_LL`, `fs`, `ks`, `f_unc`, and `k_unc`.

## Example scripts

- `python run.py` — fetch the IU.KIP LHZ Kamchatka record from IRIS and run catalog-free Dracula (3–4 mHz, global-likelihood IC counts).

## Branches

| Branch | Contents |
|---|---|
| `main` / `dev` | Current Python sources (this layout) |
| `legacy` | Earlier prototype directories, `data/`, and `timeseries-kamchatka/` |
