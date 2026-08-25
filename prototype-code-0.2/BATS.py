import numpy as np
from scipy import signal, stats

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import MCMC, NUTS, init_to_value

import math
import random

import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


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
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Bretthorst Page 17 and Eq. 3.15
    mean_square_data = jnp.sum(d ** 2) / N
    mean_square_projection = jnp.sum(h ** 2) / m

    return mean_square_data, mean_square_projection, h, eigenvalues, eigenvectors, m, N


@jax.jit
def log_probability(fs, ks, t, d):
    """Bretthorst Eq. 3.17: marginal log posterior (Student-t kernel)."""
    omegas = fs * 2.0 * jnp.pi
    msd, msp, _, _, _, m, N = _projection_quantities(t, d, omegas, ks)
    ratio = (m * msp) / (N * msd)
    return 0.5 * (m - N) * jnp.log10(1.0 - ratio)


def _bats_model_unbounded(t, d, signals):
    fs = numpyro.sample(
        "fs", 
        dist.ImproperUniform(constraints.real, batch_shape=(), event_shape=(signals,))
    )

    ks = numpyro.sample(
        "ks", 
        dist.ImproperUniform(constraints.positive, batch_shape=(), event_shape=(signals,))
    )

    numpyro.factor("bretthorst", log_probability(fs, ks, t, d))


def _bats_model_bounded(t, d, f_lo, f_hi, k_lo, k_hi):
    fs = numpyro.sample("fs", dist.Uniform(f_lo, f_hi))
    ks = numpyro.sample("ks", dist.Uniform(k_lo, k_hi))
    numpyro.factor("bretthorst", log_probability(fs, ks, t, d))


class BATS():
    def __init__(self,
                 t, 
                 d, 
                 min_f=None, 
                 max_f=None,
                 f_init=None,
                 k_init=None,
                 k_type="linear",

                 mode="global",
                 min_signals=1,
                 max_signals=10,
                 signals=1,

                 sampler="NUTS",
                 acceptance=0.8,
                 burn_in=100,
                 typical_set=200,
                 dense_mass=False,
                 boundaries=True,
                 f_bw=None,
                 k_bw=None,
                 std=5
                ):
        """
        The time series data t, d lies at the heart of BATS and what we aim to analyze. The goal of 
        BATS is to provide a (mostly) lightweight package that improves on the detection of frequencies 
        and decay rates from the FFT and other methods by using a Bayesian framework, hence Bayesian 
        Approach to Signals (BATS).

        min_f: The minimum frequency that BATS will target, informed by the Rayleigh criterion for 
               a boxcar window.
        max_f: The maximum frequency that BATS will target, informed by the Nyquist frequency.

        f_init: User specified initial frequencies if close estimates of the existing signal's frequencies
                are known.
        k_init: User specified initial decay rates if close estimates of the existing signal's decay rates
                are known.

        k_type: Expected decay of the signals. If None, then no decay will be applied.

        mode: "global" searches for the global likelihood for models with "min_signals" to "max_signals" number
              of signals. "local" will run a local likelihood for a model with "signals" signals.

        sampler: "NUTS" runs the NUTS sampler for the Hamiltonian Monte Carlo search method. If "None", then 
                 no sampling will occur, and the model statistics will be immediately returned.

        acceptance, burn_in, typical_set: "NUTS" specific parameters. See NUTS sampling to understand there uses.
        """
        # Zero the time series
        self.t = np.array(t - min(t))
        self.d = d
        self.sample_rate = np.mean(np.diff(self.t))

        # If "min_f" is not specified, then the minimum frequency is the frequency that would take 
        # one whole period to capture the entire time series.
        if min_f:
            self.min_f = min_f
        else:
            rayleigh_f = 1 / max(t)
            self.min_f = rayleigh_f

        # If "max_f" is not specified, then the maximum frequency is the Nyquist frequency to avoid 
        # aliasing.
        if max_f:
            self.max_f = max_f
        else:
            nyquist_f = 0.5 / self.sample_rate
            self.max_f = nyquist_f

        # The mode and signal numbers are required for initializing the frequencies and decay rates 
        # under certain conditions.
        self.mode = mode
        self.min_signals = min_signals
        self.max_signals = max_signals
        self.signals = signals

        if mode == "global":
            self.signals = max_signals

        self.f_init = np.zeros(self.signals)

        if f_init:
            self.f_init[:len(f_init)] = f_init
            if len(f_init) < signals or len(f_init) < max_signals:
                f, p = self._get_fft()
                f, _ = self._get_peaks(f, abs(p))

                # If initial frequencies are specified, but do not fit the number of expected 
                # frequencies, then finish populating them.
                self.f_init[len(f_init):] = f[len(f_init):self.signals]
        else:   
            f, p = self._get_fft()

        f_peaks, _ = self._get_peaks(f, np.abs(p), limit=self.signals)
        self.f_init = f_peaks

        if k_init:
            self.k_init = k_init
            if len(k_init) < self.signals or len(k_init) < max_signals:
                k = self._get_ks()
                # If initial decay rates are specified, but do not fit the number of expected 
                # decay rates, then finish populating them.
                self.k_init[len(k_init):] = k[len(k_init):self.signals]
        else:
            self.k_init = self._get_ks()

        # Finish initializing model parameters.
        self.k_type = k_type
        self.sampler = sampler
        self.acceptance = acceptance
        self.burn_in = burn_in
        self.typical_set = typical_set
        self.dense_mass = dense_mass
        self.boundaries = boundaries
        self.std = std

        if self.boundaries:
            _, _, _, uncertainties = self.get_model_statistics(self.f_init, self.k_init)
            max_f_uncertainty = jnp.nanmax(uncertainties[:int(len(uncertainties) / 2)])
            max_k_uncertainty = jnp.nanmax(uncertainties[int(len(uncertainties) / 2):])

            if f_bw is None:
                f_bw = uncertainties[:len(self.f_init)]
                self.f_bw = jnp.where(jnp.isnan(f_bw), max_f_uncertainty, f_bw)
            else:
                self.f_bw = jnp.asarray(f_bw)
                
            if k_bw is None:
                k_bw = uncertainties[:len(self.k_init)]
                self.k_bw = jnp.where(jnp.isnan(k_bw), max_k_uncertainty, k_bw)
            else:
                self.k_bw = jnp.asarray(k_bw)


    def _get_peaks(self, x, y, limit=None):
        '''
        Helper function taking the input of any 2D data set and finding its 
        peaks in order of intensity in the dependent variable y. The number of returned 
        peaks can be specified with the "limit" variable.
        '''
        peak_indices, _ = signal.find_peaks(y)

        peak_x = x[peak_indices]
        peak_y = y[peak_indices]

        sorted_indices = np.argsort(peak_y)[::-1]

        peak_x = peak_x[sorted_indices]
        peak_y = peak_y[sorted_indices]

        if limit:
            peak_x = peak_x[:limit]
            peak_y = peak_y[:limit]

        return peak_x, peak_y


    def _get_fft(self):
        '''
        Helper function for finding the FFT of BATS's provided time series using a 
        Kaiser window. Modified to splice low frequencies (<1.5MHz) from the full 
        dataset with high frequencies (>1.5MHz) from the first 2/5ths of the dataset.
        '''
        # Define the cutoff frequency for 1.5 MHz 
        # (Note: if self.sample_rate is in MHz instead of Hz, change this to 1.5)
        cutoff_freq = 1.5e6 
        
        # 1. Process the full dataset
        window_full = signal.get_window(('kaiser', 2. * np.pi), len(self.d))
        d_full = self.d * window_full
        
        # We define nfft based on the full dataset to maintain exact frequency bins
        nfft = 2 ** (math.ceil(math.log(len(self.d), 2)) + 2)
        fs = np.fft.fftfreq(n=nfft, d=self.sample_rate)
        power_full = np.fft.fft(d_full, n=nfft) * self.sample_rate

        # 2. Process the truncated dataset (first 2/5)
        idx_2_5 = int(len(self.d) * 2 / 5)
        d_trunc = self.d[:idx_2_5]
        window_trunc = signal.get_window(('kaiser', 2. * np.pi), len(d_trunc))
        d_trunc = d_trunc * window_trunc
        
        # Use the exact same nfft so the frequency bins (fs) match perfectly
        power_trunc = np.fft.fft(d_trunc, n=nfft) * self.sample_rate

        # 3. Splice the two sections together
        # Use truncated data for absolute frequencies > 1.5MHz, full data otherwise
        power_combined = np.where(np.abs(fs) > cutoff_freq, power_trunc, power_full)

        # 4. Mask the final FFT to the specified global frequency range
        mask = np.ones_like(fs, dtype=bool)
        mask &= (fs >= self.min_f) 
        mask &= (fs <= self.max_f)

        # Return the fs and powers with the mask
        return np.array(fs[mask]), np.array(power_combined[mask])

    def _get_ks(self):
        '''
        Helper function finding the initial decay rates for the initial frequencies. TODO: implement 
        decay types beyond "linear". 
        '''

        ks = np.zeros(len(self.f_init))
        
        for ind, f in enumerate(self.f_init):
            # Make a bandwidth at 99.5% - 100.5% of the selected frequency
            min_f = f * 0.995
            max_f = f * 1.005

            # Filter the data to the chosen frequency band
            order = 4
            b, a = signal.butter(order, [min_f, max_f], btype='band', fs=1 / self.sample_rate)
            detrended = signal.detrend(self.d)

            f_d = signal.filtfilt(b, a, detrended)

            peak_t, peak_d = self._get_peaks(self.t, f_d)

            valid_mask = peak_d > 0
            peak_t = peak_t[valid_mask]
            peak_d = peak_d[valid_mask]

            print(min_f, max_f)
            plt.scatter(peak_t, np.log(peak_d))
            plt.show()

            res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

            # Find the decay rate with linear regression in semilog space
            ks[ind] = -1 * res.slope

        # Return the decay rates
        return ks


    def get_model_statistics(self, fs, ks):
        """Bretthorst Eqs. 3.17, 4.7, 4.8, 4.13."""
        fs = jnp.asarray(fs, dtype=jnp.float64)
        ks = jnp.asarray(ks, dtype=jnp.float64)

        omegas = fs * 2.0 * jnp.pi
        msd, msp, h, _, _, m, N = _projection_quantities(
            self.t, self.d, omegas, ks
        )

        ratio = (m * msp) / (N * msd)
        log_prob = 0.5 * (m - N) * jnp.log(1.0 - ratio)

        # Bretthorst Eq. 4.7 and 4.8
        estimated_noise_variance = (1.0 / (N - m - 2)) * (
            jnp.sum(self.d ** 2) - jnp.sum(h ** 2)
        )
        SNR = jnp.sqrt((m / N) * (1.0 + msp / estimated_noise_variance))

        # Bretthorst Eq. 4.11 and 4.13: b matrix and parameter uncertainties.
        r = fs.shape[0]

        def lp_wrapper(q):
            return log_probability(q[:r], q[r:], self.t, self.d)

        q0 = jnp.concatenate([fs, ks])
        hessian = jax.jit(jax.hessian(lp_wrapper))(q0)

        covariance_matrix = jnp.linalg.inv(-hessian)
        parameter_uncertainties = jnp.sqrt(jnp.diag(covariance_matrix))

        return (float(log_prob), 
                float(SNR),
                float(estimated_noise_variance),
                np.asarray(parameter_uncertainties))


    def get_log_global_likelihood(self, fs, ks):
        """Bretthorst Eq. 5.9 (math preserved exactly from sandbox2.py)."""
        fs = jnp.asarray(fs, dtype=jnp.float64)
        ks = jnp.asarray(ks, dtype=jnp.float64)
        omegas = fs * 2.0 * jnp.pi

        msd, msp, _, _, _, m, N = _projection_quantities(
            self.t, self.d, omegas, ks
        )
        r = fs.shape[0]

        R_delta = float(jnp.max(jnp.abs(self.d)))
        R_sigma = float(jnp.max(jnp.abs(self.d)))

        log_R_delta = jnp.maximum(jnp.log(R_delta), 1e-12)
        log_R_sigma = jnp.maximum(jnp.log(R_sigma), 1e-12)

        R_gamma = (0.5 / float(jnp.mean(jnp.diff(self.t)))) * float(self.t[-1] - self.t[0])

        def msp_of_q(q):
            f = q[:r]
            a = q[r:]
            o = f * 2.0 * jnp.pi
            _, msp_in, _, _, _, _, _ = _projection_quantities(self.t, self.d, o, a)
            return msp_in

        q0 = jnp.concatenate([fs, ks])
        hess = jax.jit(jax.hessian(msp_of_q))(q0)
        b = (-m / 2.0) * hess
        eigenvalues, _ = jnp.linalg.eigh(b)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        factor = ((m / 2.0) * jnp.log(2.0 * jnp.pi)
                    - 0.5 * jnp.sum(jnp.log(eigenvalues))
                    - m * jnp.log(R_gamma))
        delta_term = (jsp.gammaln(m / 2.0)
                        - jnp.log(2.0 * log_R_delta)
                        - (m / 2.0) * jnp.log((m * msp) / 2.0))
        sigma_term = (jsp.gammaln((N - m - r) / 2.0)
                        - jnp.log(2.0 * log_R_sigma)
                        - ((N - m - r) / 2.0) * jnp.log((N * msd - m * msp) / 2.0))
        gamma_term = -(2.0 * r) * jnp.log(R_gamma)

        return float(delta_term + sigma_term + gamma_term + factor)


    def nuts_sampler_unbounded(self):
        init_dict = {"fs": self.f_init, "ks": self.k_init}
        init_strategy = init_to_value(values=init_dict)

        kernel = NUTS(_bats_model_unbounded,
                      init_strategy=init_strategy,
                      dense_mass=self.dense_mass,
                      target_accept_prob=self.acceptance,
                      max_tree_depth=5)
        mcmc = MCMC(kernel,
                    num_warmup=self.burn_in,
                    num_samples=self.typical_set,
                    num_chains=1,
                    progress_bar=True)

        seed = 0
            
        mcmc.run(jax.random.PRNGKey(seed),
                 self.t, 
                 self.d,
                 3)
    
        samples = mcmc.get_samples()


    def nuts_sampler_bounded(self, fs, ks, f_bw, k_bw):
        f_lo = fs - (self.std * f_bw)
        f_hi = fs + (self.std * f_bw)

        print(ks, k_bw, self.std)
        k_lo = jnp.array(ks) - (self.std * jnp.array(k_bw))
        k_hi = jnp.array(ks) + (self.std * jnp.array(k_bw))

        k_lo = jnp.maximum(k_lo, 0)
        k_hi = jnp.maximum(k_hi, 1e-6)
        k_init = jnp.maximum(jnp.array(ks), 1e-12)

        eps_f = 1e-6 * f_bw
        eps_k = 1e-6 * k_bw

        f_init = jnp.clip(fs, f_lo + eps_f, f_hi - eps_f)
        k_init = jnp.clip(ks, k_lo + eps_k, k_hi - eps_k)

        init_strategy = init_to_value(values={"fs": f_init, "ks": k_init})

        kernel = NUTS(_bats_model_bounded,
                      init_strategy=init_strategy,
                      dense_mass=self.dense_mass,
                      target_accept_prob=self.acceptance,
                      max_tree_depth=8)
        mcmc = MCMC(kernel,
                    num_warmup=self.burn_in,
                    num_samples=self.typical_set,
                    num_chains=1,
                    progress_bar=True)

        seed = 0

        mcmc.run(jax.random.PRNGKey(seed),
                 self.t, self.d,
                 f_lo, f_hi, k_lo, k_hi,
                 extra_fields=("potential_energy",))

        samples = mcmc.get_samples()

        # last_state = mcmc.last_state
        # inv_mass_matrix = last_state.adapt_state.inverse_mass_matrix
        # mass_sqrt = last_state.adapt_state.mass_matrix_sqrt

        pe = mcmc.get_extra_fields()["potential_energy"]
        best_index = jnp.argmin(pe)

        fs = samples["fs"][best_index]
        ks = samples["ks"][best_index]

        return fs, ks
         

    def run_global_model(self):
        global_likelihoods = np.zeros(self.max_signals - self.min_signals + 1)

        for signals in range(self.min_signals, self.max_signals + 1):
            fs, ks = self.run_local_model(signals)

            global_likelihoods[signals - self.min_signals] = self.get_log_global_likelihood(fs, ks)


    def run_local_model(self, signals):
        if self.sampler == "NUTS":
            if self.boundaries == True:
                fs, ks = self.nuts_sampler_bounded(self.f_init[:signals], 
                                                   self.k_init[:signals], 
                                                   self.f_bw[:signals], 
                                                   self.k_bw[:signals])
            else:
                fs, ks = self.nuts_sampler_unbounded()

        _, SNR, variance, _ = self.get_model_statistics(fs, ks)

        header_metadata = (
            f"SNR: {SNR:.4f} | Noise Variance: {variance:.6f}\n"
            f"frequency,decay_rate"
        )

        sort_indices = np.argsort(fs)
        fs = fs[sort_indices]
        ks = ks[sort_indices]

        output_data = np.column_stack((fs, ks))

        np.savetxt(
            f"{signals}_model.csv", 
            output_data, 
            delimiter=",", 
            header=header_metadata, 
            comments=""
        )

        return fs, ks


    def launch(self):
        if self.mode == "local":
            self.run_local_model(self.signals)

        if self.mode == "global":
            self.run_global_model()




if __name__ == "__main__":
    rng = np.random.default_rng()

    a_1 = 2
    a_2 = 3
    a_3 = 4
    a_4 = 3

    f_1 = 4
    f_2 = 5
    f_3 = 8.95
    f_4 = 9.05

    k_1 = 0.010
    k_2 = 0.005
    k_3 = 0.020
    k_4 = 0.025

    t = np.linspace(0, 1_000, 30_000)

    e = rng.uniform(low=-1, high=1, size=30_000)
    d = (a_1 * np.sin(2 * np.pi * f_1 * t) * np.e ** (-k_1 * t) + 
         a_2 * np.sin(2 * np.pi * f_2 * t) * np.e ** (-k_2 * t) + 
         a_3 * np.sin(2 * np.pi * f_3 * t) * np.e ** (-k_3 * t) +
         a_4 * np.sin(2 * np.pi * f_4 * t) * np.e ** (-k_4 * t) + e)

    plt.plot(t, d)
    plt.show()

    model = BATS(t, 
                 d, 
                 f_init=[4, 5, 8.95, 9.05],
                 mode="global", 
                 min_signals=1, 
                 max_signals=5, 
                 burn_in=10,
                 typical_set=20,
                 boundaries=True, 
                 std=5,
                 k_bw=[0.04, 0.04, 0.04, 0.04, 0.04])
    model.launch()
