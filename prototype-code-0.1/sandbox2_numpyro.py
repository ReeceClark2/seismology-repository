"""BATS - Bayesian Approach to Signals (NumPyro/NUTS rewrite of sandbox2.py).

Same Bretthorst (1988) math as sandbox2.py, but with:
- Single shared orthonormal-projection helper (no duplicated Gram-matrix code).
- End-to-end JAX/JIT compilation (no host-device transfers in the inner loop).
- NumPyro NUTS sampler with adaptive step size and dense mass-matrix warmup.
- Bug fixes: self.scale always defined; input array d never mutated in place.

Public interface preserved where reasonable. L and epsilon are now ignored (NUTS
adapts both). B -> num_warmup, M -> num_samples.
"""

from __future__ import annotations

import csv
import math
import time
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.infer.initialization import init_to_value

from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


# ============================================================
# Shared Bretthorst orthonormal projection
# ============================================================

def _projection_quantities(t, d, omegas, alphas):
    """Bretthorst orthonormal projection. Single source of truth.

    Returns (mean_square_data, mean_square_projection, h, eigenvalues,
             eigenvectors, m, N).
    """
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Bretthorst Page 17 and Eq. 3.15
    mean_square_data = jnp.sum(d ** 2) / N
    mean_square_projection = jnp.sum(h ** 2) / m

    return mean_square_data, mean_square_projection, h, eigenvalues, eigenvectors, m, N


@jax.jit
def log_probability(frequencies, decay_rates, t, d):
    """Bretthorst Eq. 3.17: marginal log posterior (Student-t kernel)."""
    omegas = frequencies * 2.0 * jnp.pi
    msd, msp, _, _, _, m, N = _projection_quantities(t, d, omegas, decay_rates)
    ratio = (m * msp) / (N * msd)
    return 0.5 * (m - N) * jnp.log10(1.0 - ratio)


# ============================================================
# NumPyro probabilistic model
# ============================================================

def _bats_model(t, d, freq_lower, freq_upper, decay_lower, decay_upper):
    freqs = numpyro.sample("freqs", dist.Uniform(freq_lower, freq_upper))
    decays = numpyro.sample("decays", dist.Uniform(decay_lower, decay_upper))
    numpyro.factor("bretthorst", log_probability(freqs, decays, t, d))


# ============================================================
# BATS class
# ============================================================

class BATS:
    def __init__(self,
                 t,
                 d,
                 minimum_frequency=0,
                 maximum_frequency=None,
                 signals=None,
                 guess_frequencies=None,
                 guess_decay_rates=None,
                 decay_type="lorentzian"):
        t_arr = np.asarray(t, dtype=np.float64)
        d_arr = np.asarray(d, dtype=np.float64).copy()  # never mutate caller

        d_max = float(np.max(np.abs(d_arr)))
        self.scale = 2.72 / d_max if d_max < 2.72 else 1.0
        d_arr *= self.scale

        self.t = jnp.asarray(t_arr)
        self.d = jnp.asarray(d_arr)

        self.minimum_frequency = minimum_frequency
        if maximum_frequency is None:
            self.maximum_frequency = 0.5 / float(np.mean(np.diff(t_arr)))
        else:
            self.maximum_frequency = maximum_frequency

        if guess_frequencies is None:
            f, p = self._get_fft(t_arr, d_arr,
                                 self.minimum_frequency, self.maximum_frequency)
            peaks_f, _ = self._get_peaks(f, np.abs(p))
            self.guess_frequencies = peaks_f.tolist()
        else:
            self.guess_frequencies = list(guess_frequencies)

        if guess_decay_rates is None:
            decays = self._get_decay_rates(t_arr, d_arr, self.guess_frequencies)
            self.guess_decay_rates = [max(0.0, x) for x in decays]
        else:
            self.guess_decay_rates = list(guess_decay_rates)

        if signals is None:
            self.signals = len(self.guess_frequencies)
        else:
            self.signals = signals
            self.guess_frequencies = self.guess_frequencies[:signals]
            self.guess_decay_rates = self.guess_decay_rates[:signals]

        self.decay_type = decay_type

    @staticmethod
    def _get_fft(t, d, minimum_frequency=None, maximum_frequency=None):
        delta = np.mean(np.diff(t))
        taper = sp_signal.get_window(('kaiser', 2.0 * np.pi), len(d))
        d_windowed = d * taper
        nfft = 2 ** (math.ceil(math.log2(len(d))) + 4)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d_windowed, n=nfft) * delta
        mask = np.ones_like(frequencies, dtype=bool)
        if minimum_frequency is not None:
            mask &= frequencies >= minimum_frequency
        if maximum_frequency is not None:
            mask &= frequencies <= maximum_frequency
        return frequencies[mask], power[mask]

    @staticmethod
    def _get_peaks(x, y):
        deltas = np.diff(y)
        peak_indices = np.where((deltas[:-1] > 0) & (deltas[1:] < 0))[0] + 1
        peak_x = x[peak_indices]
        peak_y = y[peak_indices]
        order = np.argsort(peak_y)[::-1]
        return peak_x[order], peak_y[order]

    @staticmethod
    def _get_decay_rates(t, d, frequencies):
        delta = np.mean(np.diff(t))
        fs = 1.0 / delta
        nyquist = 0.5 * fs
        decay_rates = np.zeros(len(frequencies))
        for ind, freq in enumerate(frequencies):
            low = (freq * 0.995) / nyquist
            high = (freq * 1.005) / nyquist
            b, a = sp_signal.butter(4, [low, high], btype='band')
            detrended = sp_signal.detrend(d)
            filtered = sp_signal.filtfilt(b, a, detrended)
            peak_t, peak_d = BATS._get_peaks(t, filtered)
            res = sp_stats.linregress(peak_t, np.log(peak_d))
            decay_rates[ind] = -res.slope
        return decay_rates.tolist()

    def nuts_sample(self,
                    guess_frequencies,
                    guess_decay_rates,
                    num_warmup=1_000,
                    num_samples=2_000,
                    freq_bandwidth=1e-6,
                    decay_bandwidth=3e-6,
                    target_accept_prob=0.65,
                    dense_mass=False,
                    max_tree_depth=5,
                    seed=0):    
        """Run NUTS sampling on the BATS model.

        max_tree_depth=5 caps NUTS at 2^5 = 32 leapfrog steps per sample,
        roughly matching the original sandbox2.py's fixed L=30 per sample.
        Default NumPyro max_tree_depth is 10 (up to 1024 steps), which makes
        the sampler much slower than fixed-L HMC for our tight-prior geometry.
        """
        f0 = jnp.asarray(guess_frequencies, dtype=jnp.float64)
        a0 = jnp.asarray(guess_decay_rates, dtype=jnp.float64)
        freq_lower = f0 - freq_bandwidth
        freq_upper = f0 + freq_bandwidth
        decay_lower = jnp.maximum(0.0, a0 - decay_bandwidth)
        decay_upper = a0 + decay_bandwidth

        # Start chain at the user's guess (clipped slightly inside the support
        # so the unconstraining transform is finite).
        eps = 1e-3 * freq_bandwidth
        f_init = jnp.clip(f0, freq_lower + eps, freq_upper - eps)
        a_init = jnp.clip(a0, decay_lower + eps, decay_upper - eps)
        init_strategy = init_to_value(values={"freqs": f_init, "decays": a_init})

        kernel = NUTS(_bats_model,
                      init_strategy=init_strategy,
                      dense_mass=dense_mass,
                      target_accept_prob=target_accept_prob,
                      max_tree_depth=max_tree_depth)
        mcmc = MCMC(kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=1,
                    progress_bar=True)
        mcmc.run(jax.random.PRNGKey(seed),
                 self.t, self.d,
                 freq_lower, freq_upper, decay_lower, decay_upper,
                 extra_fields=("num_steps", "diverging", "accept_prob"))

        samples = mcmc.get_samples()
        extras = mcmc.get_extra_fields()

        # Evaluate the Bretthorst log probability at each sample.
        log_probs = jax.vmap(
            lambda f, a: log_probability(f, a, self.t, self.d)
        )(samples["freqs"], samples["decays"])

        result = {
            "freqs": np.asarray(samples["freqs"]),
            "decays": np.asarray(samples["decays"]),
            "log_prob": np.asarray(log_probs),
            "num_steps": np.asarray(extras["num_steps"]),
            "diverging": np.asarray(extras["diverging"]),
            "accept_prob": np.asarray(extras["accept_prob"]),
        }
        return result, mcmc

    def get_model_statistics(self, frequencies, decay_rates):
        """Bretthorst Eqs. 3.17, 4.7, 4.8, 4.13."""
        frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
        decay_rates = jnp.asarray(decay_rates, dtype=jnp.float64)
        d_unscaled = self.d / self.scale

        omegas = frequencies * 2.0 * jnp.pi
        msd, msp, h, _, _, m, N = _projection_quantities(
            self.t, d_unscaled, omegas, decay_rates
        )

        ratio = (m * msp) / (N * msd)
        log_prob = 0.5 * (m - N) * jnp.log10(1.0 - ratio)

        # Bretthorst Eq. 4.7 and 4.8
        estimated_noise_variance = (1.0 / (N - m - 2)) * (
            jnp.sum(d_unscaled ** 2) - jnp.sum(h ** 2)
        )
        SNR = jnp.sqrt((m / N) * (1.0 + msp / estimated_noise_variance))

        # Bretthorst Eq. 4.11 and 4.13: b matrix and parameter uncertainties.
        r = frequencies.shape[0]

        def lp_wrapper(q):
            return log_probability(q[:r], q[r:], self.t, d_unscaled)

        q0 = jnp.concatenate([frequencies, decay_rates])
        hess = jax.jit(jax.hessian(lp_wrapper))(q0)
        b = (-m / 2.0) * hess
        eigenvalues, eigenvectors = jnp.linalg.eigh(b)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)
        parameter_uncertainties = jnp.sqrt(
            estimated_noise_variance
            * jnp.sum((eigenvectors ** 2) / eigenvalues, axis=1)
        )

        return (float(log_prob), float(SNR),
                float(estimated_noise_variance),
                np.asarray(parameter_uncertainties))

    def get_log_global_likelihood(self, frequencies, decay_rates):
        """Bretthorst Eq. 5.9 (math preserved exactly from sandbox2.py)."""
        frequencies = jnp.asarray(frequencies, dtype=jnp.float64)
        decay_rates = jnp.asarray(decay_rates, dtype=jnp.float64)
        omegas = frequencies * 2.0 * jnp.pi

        msd, msp, _, _, _, m, N = _projection_quantities(
            self.t, self.d, omegas, decay_rates
        )
        r = frequencies.shape[0]

        R_delta = float(jnp.max(jnp.abs(self.d)))
        R_sigma = float(jnp.max(jnp.abs(self.d)))
        R_gamma = (0.5 / float(jnp.mean(jnp.diff(self.t)))) * float(self.t[-1] - self.t[0])

        def msp_of_q(q):
            f = q[:r]
            a = q[r:]
            o = f * 2.0 * jnp.pi
            _, msp_in, _, _, _, _, _ = _projection_quantities(self.t, self.d, o, a)
            return msp_in

        q0 = jnp.concatenate([frequencies, decay_rates])
        hess = jax.jit(jax.hessian(msp_of_q))(q0)
        b = (-m / 2.0) * hess
        eigenvalues, _ = jnp.linalg.eigh(b)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        factor = ((m / 2.0) * jnp.log(2.0 * jnp.pi)
                  - 0.5 * jnp.sum(jnp.log(eigenvalues))
                  - m * jnp.log(R_gamma))
        delta_term = (jsp.gammaln(m / 2.0)
                      - jnp.log(2.0 * jnp.log(R_delta))
                      - (m / 2.0) * jnp.log((m * msp) / 2.0))
        sigma_term = (jsp.gammaln((N - m - r) / 2.0)
                      - jnp.log(2.0 * jnp.log(R_sigma))
                      - ((N - m - r) / 2.0) * jnp.log((N * msd - m * msp) / 2.0))
        gamma_term = -(2.0 * r) * jnp.log(R_gamma)

        return float(delta_term + sigma_term + gamma_term + factor)

    def run(self,
            lower_bound_model_functions=1,
            upper_bound_model_functions=5,
            B=1_000,
            M=2_000,
            L=None,          # ignored (NUTS adapts trajectory length)
            epsilon=None,    # ignored (NUTS adapts step size)
            freq_bandwidth=1e-6,
            decay_bandwidth=5e-6,
            output_file="BATS_model_results.txt"):
        if L is not None or epsilon is not None:
            print("Note: L and epsilon are ignored by the NUTS sampler.")

        n_models = upper_bound_model_functions - lower_bound_model_functions + 1
        per_model_samples = []
        per_model_best = []
        per_model_uncert = []
        global_likelihoods = np.zeros(n_models)
        SNRs = np.zeros(n_models)
        variances = np.zeros(n_models)
        accept_rates = np.zeros(n_models)
        divergence_rates = np.zeros(n_models)

        for i in range(n_models):
            r = lower_bound_model_functions + i
            print(f"\nFitting {r}-function model with NUTS...")

            gf = self.guess_frequencies[:r]
            ga = self.guess_decay_rates[:r]

            t0 = time.perf_counter()
            samples, _ = self.nuts_sample(
                gf, ga,
                num_warmup=B, num_samples=M,
                freq_bandwidth=freq_bandwidth,
                decay_bandwidth=decay_bandwidth,
                seed=i,
            )
            t_elapsed = time.perf_counter() - t0
            print(f"  NUTS finished in {t_elapsed:.1f}s "
                  f"(mean accept_prob={samples['accept_prob'].mean():.3f}, "
                  f"divergences={int(samples['diverging'].sum())}, "
                  f"mean tree leaves={samples['num_steps'].mean():.1f})")

            per_model_samples.append(samples)
            accept_rates[i] = 100.0 * float(samples["accept_prob"].mean())
            divergence_rates[i] = 100.0 * float(samples["diverging"].mean())

            best_idx = int(np.argmax(samples["log_prob"]))
            best_f = samples["freqs"][best_idx]
            best_a = samples["decays"][best_idx]
            per_model_best.append((best_f, best_a))

            _, snr, var, unc = self.get_model_statistics(best_f, best_a)

            SNRs[i] = snr
            variances[i] = var
            per_model_uncert.append(unc)

            global_likelihoods[i] = self.get_log_global_likelihood(best_f, best_a)

            # Write per-model sample CSV.
            self._write_samples_csv(samples, r)

        self._write_summary(output_file,
                            lower_bound_model_functions,
                            per_model_best,
                            per_model_uncert,
                            global_likelihoods,
                            SNRs, variances,
                            accept_rates, divergence_rates)

        return per_model_samples, per_model_best, global_likelihoods, SNRs, variances

    @staticmethod
    def _write_samples_csv(samples, r):
        path = f"BATS_model_results_{r}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["iteration", "function", "frequency",
                             "decay_rate", "probability"])
            for it in range(samples["freqs"].shape[0]):
                lp = samples["log_prob"][it]
                for j in range(r):
                    writer.writerow([it, j,
                                     samples["freqs"][it, j],
                                     samples["decays"][it, j],
                                     lp])

    @staticmethod
    def _write_summary(path, lower, bests, uncerts, log_ls,
                       snrs, variances, accept, diverge):
        max_log_l = float(np.nanmax(log_ls))
        exp_terms = np.exp(log_ls - max_log_l)
        probs = exp_terms / np.nansum(exp_terms)
        best_index = int(np.nanargmax(log_ls))

        with open(path, "w", encoding="utf-8") as fh:
            for ind, ((bf, ba), unc) in enumerate(zip(bests, uncerts)):
                r = len(bf)
                freq_unc = unc[:r]
                decay_unc = unc[r:]
                print(f"\nDetails for {ind + lower}-frequency model:", file=fh)
                print(f"{'Signal':<8} | {'Frequency +- Uncertainty':<32} | {'Decay Rate +- Uncertainty':<36}", file=fh)
                print("-" * 90, file=fh)
                for i in range(r):
                    print(f"{i+1:<8} | "
                          f"{bf[i]:>14.10f} +- {freq_unc[i]:<14.10f} | "
                          f"{ba[i]:>16.12f} +- {decay_unc[i]:<16.12f}",
                          file=fh)
                print("-" * 90, file=fh)

            print(f"\n{'Order (r)':<10} | {'Log-Likelihood':<18} | "
                  f"{'SNR':<10} | {'Variance':<12} | {'Probability':<14} | "
                  f"{'Accept %':<10} | {'Diverge %':<10}", file=fh)
            print("-" * 100, file=fh)
            for i, log_l in enumerate(log_ls):
                print(f"{i + lower:<10} | {log_l:<18.2f} | "
                      f"{snrs[i]:<10.2f} | {np.log(variances[i]):<12.2f} | "
                      f"{probs[i]:<14.4e} | "
                      f"{accept[i]:<10.4f} | {diverge[i]:<10.4f}",
                      file=fh)
            print("-" * 100, file=fh)
            print(f"The best model is the {best_index + lower}-frequency model!",
                  file=fh)


# ============================================================
# Data loaders (unchanged from sandbox2.py)
# ============================================================

def get_synthetic_data(filename, minimum_frequency=None, maximum_frequency=None):
    data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')
    t = data.iloc[:, 0].values
    d = data.iloc[:, 1].values

    delta = np.mean(np.diff(t))
    fs = 1.0 / delta
    nyquist = 0.5 * fs
    low = minimum_frequency / nyquist
    high = maximum_frequency / nyquist

    b, a = sp_signal.butter(4, [low, high], btype='band')
    detrended = sp_signal.detrend(d)
    d = sp_signal.filtfilt(b, a, detrended)

    t = t[144:]
    d = d[144:]
    return t, d


def get_observed_data(network, station, channel, location, stream_index,
                      start_time, end_time, minimum_frequency, maximum_frequency):
    client = Client('IRIS')
    inventory = client.get_stations(network=network, station=station,
                                    location=location, channel=channel,
                                    starttime=start_time, endtime=end_time,
                                    level='response')
    stream = client.get_waveforms(network=network, station=station,
                                  location=location, channel=channel,
                                  starttime=start_time, endtime=end_time)
    trace = stream[stream_index]
    trace.detrend('constant')
    trace.remove_response(inventory=inventory, output="ACC")
    if minimum_frequency and maximum_frequency:
        trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(5, no_filter=False)
    trace.decimate(2, no_filter=False)

    delta = trace.stats.delta
    N = len(trace)
    t = np.arange(N) * delta
    d = np.array(trace.data)
    t = t[144:]
    d = d[144:]
    return t, d


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    minimum_frequency = 0.00025
    maximum_frequency = 0.00130

    network = "IU"
    station = "KIP"
    channel = "LHZ"
    location = "00"

    stream_index = 0
    start_time = UTCDateTime('2025-07-31T06:24:50')
    end_time = UTCDateTime('2025-08-11T05:24:50')

    file_path = f"../timeseries-kamchatka/{network}_{station}_TS.ascii"

    lower_bound_model_functions = 40
    upper_bound_model_functions = 40

    B = 100   # num_warmup
    M = 300   # num_samples

    data = "observed"
    # -------------------------------------------------------------------------

    if data == "observed":
        t, d = get_observed_data(network, station, channel, location,
                                 stream_index, start_time, end_time,
                                 minimum_frequency, maximum_frequency)
    else:
        t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)

    model = BATS(
        t, d,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )

    t_start = time.perf_counter()
    model.run(lower_bound_model_functions=lower_bound_model_functions,
              upper_bound_model_functions=upper_bound_model_functions,
              B=B, M=M)
    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f}s")
