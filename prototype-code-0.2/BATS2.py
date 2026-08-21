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
import concurrent.futures

jax.config.update("jax_enable_x64", True)


def _projection_quantities(t, d, omegas, alphas):
    """Bretthorst orthonormal projection. Single source of truth."""
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-alphas[:, None] * t[None, :])

    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    h = H @ d

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


def _parallel_worker(kwargs):
    """Isolated worker that instantiates a local BATS model and runs it."""
    model = BATS(**kwargs)
    return model.run_local_model()


class BATS():
    def __init__(self, t, d, min_f=None, max_f=None, f_init=None, k_init=None,
                 k_type="linear", mode="global", min_signals=1, max_signals=10,
                 signals=1, sampler="NUTS", acceptance=0.8, burn_in=100,
                 typical_set=200, dense_mass=False, boundaries=True,
                 f_bw=None, k_bw=None, std=5, processes=1, _chunk_idx=None, 
                 _is_worker=False):
        
        self.t = np.array(t - min(t))
        self.d = d
        self.sample_rate = np.mean(np.diff(self.t))

        self.min_f = min_f if min_f else 1 / max(t)
        self.max_f = max_f if max_f else 0.5 / self.sample_rate

        self.f_init = np.asarray(f_init) if f_init is not None else np.array([])
        self.k_init = np.asarray(k_init) if k_init is not None else np.array([])
        
        self.k_type = k_type
        self.mode = mode
        self.min_signals = min_signals
        self.max_signals = max_signals
        self.signals = signals if f_init is None else len(self.f_init)

        self.sampler = sampler
        self.acceptance = acceptance
        self.burn_in = burn_in
        self.typical_set = typical_set
        self.dense_mass = dense_mass
        self.boundaries = boundaries
        self.std = std
        self.processes = processes
        self._chunk_idx = _chunk_idx
        self._is_worker = _is_worker

        # If this is a spawned worker with signals, prep boundaries
        if self._is_worker and self.signals > 0:
            if len(self.k_init) < self.signals:
                self.k_init = self._get_ks()

            if self.boundaries:
                _, _, _, uncertainties = self.get_model_statistics(self.f_init, self.k_init)
                max_f_uncertainty = jnp.nanmax(uncertainties[:int(len(uncertainties) / 2)])
                max_k_uncertainty = jnp.nanmax(uncertainties[int(len(uncertainties) / 2):])

                if f_bw is None:
                    fb = uncertainties[:len(self.f_init)]
                    self.f_bw = jnp.where(jnp.isnan(fb), max_f_uncertainty, fb)
                else:
                    self.f_bw = jnp.asarray(f_bw)
                    
                if k_bw is None:
                    kb = uncertainties[len(self.f_init):]
                    self.k_bw = jnp.where(jnp.isnan(kb), max_k_uncertainty, kb)
                else:
                    self.k_bw = jnp.asarray(k_bw)

    def _get_peaks(self, x, y, limit=None):
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
        window = signal.get_window(('kaiser', 2. * np.pi), len(self.d))
        d_win = self.d * window
        nfft = 2 ** (math.ceil(math.log(len(self.d), 2)) + 2)
        fs = np.fft.fftfreq(n=nfft, d=self.sample_rate)
        power = np.fft.fft(d_win, n=nfft) * self.sample_rate
        mask = (fs >= self.min_f) & (fs <= self.max_f)
        return np.array(fs[mask]), np.array(power[mask])
    
    def _get_ks(self):
        ks = np.zeros(len(self.f_init))
        for ind, f in enumerate(self.f_init):
            if f <= 0:
                ks[ind] = 1e-4
                continue
            min_f, max_f = f * 0.995, f * 1.005
            try:
                b, a = signal.butter(4, [min_f, max_f], btype='band', fs=1 / self.sample_rate)
                detrended = signal.detrend(self.d)
                f_d = signal.filtfilt(b, a, detrended)
                peak_t, peak_d = self._get_peaks(self.t, f_d)
                valid_mask = peak_d > 0
                peak_t, peak_d = peak_t[valid_mask], peak_d[valid_mask]

                if len(peak_t) > 1:
                    res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))
                    ks[ind] = max(-1 * res.slope, 1e-6) 
                else:
                    ks[ind] = 1e-4
            except ValueError:
                ks[ind] = 1e-4
        return ks

    def get_model_statistics(self, fs, ks):
        fs = jnp.asarray(fs, dtype=jnp.float64)
        ks = jnp.asarray(ks, dtype=jnp.float64)
        omegas = fs * 2.0 * jnp.pi
        msd, msp, h, _, _, m, N = _projection_quantities(self.t, self.d, omegas, ks)
        ratio = (m * msp) / (N * msd)
        log_prob = 0.5 * (m - N) * jnp.log(1.0 - ratio)
        estimated_noise_variance = (1.0 / (N - m - 2)) * (jnp.sum(self.d ** 2) - jnp.sum(h ** 2))
        SNR = jnp.sqrt((m / N) * (1.0 + msp / estimated_noise_variance))
        
        r = fs.shape[0]
        def lp_wrapper(q):
            return log_probability(q[:r], q[r:], self.t, self.d)
        q0 = jnp.concatenate([fs, ks])
        hessian = jax.jit(jax.hessian(lp_wrapper))(q0)
        covariance_matrix = jnp.linalg.inv(-hessian)
        parameter_uncertainties = jnp.sqrt(jnp.diag(covariance_matrix))
        return float(log_prob), float(SNR), float(estimated_noise_variance), np.asarray(parameter_uncertainties)

    def nuts_sampler_unbounded(self):
        f_init_safe = jnp.asarray(self.f_init)
        k_init_safe = jnp.maximum(jnp.asarray(self.k_init), 1e-6)
        init_strategy = init_to_value(values={"fs": f_init_safe, "ks": k_init_safe})
        kernel = NUTS(_bats_model_unbounded, init_strategy=init_strategy, dense_mass=self.dense_mass, target_accept_prob=self.acceptance, max_tree_depth=5)
        mcmc = MCMC(kernel, num_warmup=self.burn_in, num_samples=self.typical_set, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), self.t, self.d, len(self.f_init), extra_fields=("potential_energy",))
        samples = mcmc.get_samples()
        best_index = jnp.argmin(mcmc.get_extra_fields()["potential_energy"])
        return samples["fs"][best_index], samples["ks"][best_index]

    def nuts_sampler_bounded(self, fs, ks, f_bw, k_bw):
        f_lo, f_hi = fs - (self.std * f_bw), fs + (self.std * f_bw)
        k_lo, k_hi = jnp.maximum(ks - (self.std * k_bw), 0), jnp.maximum(ks + (self.std * k_bw), 1e-6)
        eps_f, eps_k = 1e-6 * f_bw, 1e-6 * k_bw
        f_init, k_init = jnp.clip(fs, f_lo + eps_f, f_hi - eps_f), jnp.clip(ks, k_lo + eps_k, k_hi - eps_k)
        
        init_strategy = init_to_value(values={"fs": f_init, "ks": k_init})
        kernel = NUTS(_bats_model_bounded, init_strategy=init_strategy, dense_mass=self.dense_mass, target_accept_prob=self.acceptance, max_tree_depth=8)
        mcmc = MCMC(kernel, num_warmup=self.burn_in, num_samples=self.typical_set, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), self.t, self.d, f_lo, f_hi, k_lo, k_hi, extra_fields=("potential_energy",))
        samples = mcmc.get_samples()
        best_index = jnp.argmin(mcmc.get_extra_fields()["potential_energy"])
        return samples["fs"][best_index], samples["ks"][best_index]

    def run_local_model(self):
        """Worker execution method. Returns data instead of saving."""
        if self.signals == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0.0, 0.0, 0.0, self._chunk_idx

        if self.sampler == "NUTS":
            if self.boundaries:
                fs, ks = self.nuts_sampler_bounded(self.f_init, self.k_init, self.f_bw, self.k_bw)
            else:
                fs, ks = self.nuts_sampler_unbounded()
        
        log_p, SNR, variance, uncertainties = self.get_model_statistics(fs, ks)
        f_unc, k_unc = uncertainties[:len(fs)], uncertainties[len(fs):]
        
        sort_indices = np.argsort(fs)
        return fs[sort_indices], ks[sort_indices], f_unc[sort_indices], k_unc[sort_indices], log_p, SNR, variance, self._chunk_idx

    def _consolidate_and_save(self, total_signals, chunk_cache):
        """Merges results from all chunks and saves a single CSV."""
        all_fs, all_ks, all_func, all_kunc, all_chunks = [], [], [], [], []
        global_snr, global_var, global_log_p = 0.0, 0.0, 0.0
        active_chunks = 0
        
        for c_idx, data in chunk_cache.items():
            if data is None: continue
            fs, ks, func, kunc, log_p, snr, var, _ = data
            if len(fs) > 0:
                all_fs.extend(fs)
                all_ks.extend(ks)
                all_func.extend(func)
                all_kunc.extend(kunc)
                all_chunks.extend([c_idx] * len(fs))
                global_log_p += log_p
                global_snr += snr
                global_var += var
                active_chunks += 1

        if active_chunks > 0:
            global_snr /= active_chunks
            global_var /= active_chunks

        header = f"Total Log-Prob: {global_log_p:.4f} | Avg SNR: {global_snr:.4f} | Avg Variance: {global_var:.6e}\nfrequency,decay_rate,f_uncertainty,k_uncertainty,chunk_id"
        
        if len(all_fs) > 0:
            output_data = np.column_stack((all_fs, all_ks, all_func, all_kunc, all_chunks)).astype(float)
            # Sort globally by frequency
            output_data = output_data[output_data[:, 0].argsort()]
        else:
            output_data = np.empty((0, 5))

        np.savetxt(f"{total_signals}_model_consolidated.csv", output_data, delimiter=",", header=header, comments="")

    def launch(self):
        """Orchestrator: extracts global priors, assigns to chunks, intelligently caches execution."""
        if self.mode == "local":
            print("Local mode runs without chunking optimizations. Use global mode for caching behavior.")
            self._is_worker = True
            self.run_local_model()
            return

        # 1. Global Peak Finding
        f_vals, p_vals = self._get_fft()
        global_priors_f, _ = self._get_peaks(f_vals, abs(p_vals), limit=self.max_signals)
        
        # 2. Setup Chunks
        freq_bins = np.linspace(self.min_f, self.max_f, self.processes + 1)
        
        # Dictionary to store the current configuration of signals per chunk
        current_chunk_priors = {i: [] for i in range(self.processes)}
        chunk_cache = {i: None for i in range(self.processes)}

        executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.processes)

        for signals in range(self.min_signals, self.max_signals + 1):
            target_priors = global_priors_f[:signals]
            
            # 3. Bin priors into chunks
            new_chunk_priors = {i: [] for i in range(self.processes)}
            bin_indices = np.digitize(target_priors, freq_bins) - 1
            for idx, freq in zip(bin_indices, target_priors):
                # Handle edge cases where digitize places it in the last bin+1
                idx = min(max(idx, 0), self.processes - 1)
                new_chunk_priors[idx].append(freq)

            # 4. Determine which chunks need re-running
            tasks_to_run = []
            for i in range(self.processes):
                if sorted(new_chunk_priors[i]) != sorted(current_chunk_priors[i]) or chunk_cache[i] is None:
                    # Configuration changed (or first run), dispatch to worker
                    kwargs = {
                        't': self.t, 'd': self.d, 'min_f': freq_bins[i], 'max_f': freq_bins[i+1],
                        'f_init': new_chunk_priors[i], 'k_init': None, 'sampler': self.sampler,
                        'boundaries': self.boundaries, 'std': self.std, 'acceptance': self.acceptance,
                        'burn_in': self.burn_in, 'typical_set': self.typical_set,
                        '_chunk_idx': i, '_is_worker': True, 'processes': 1
                    }
                    tasks_to_run.append(executor.submit(_parallel_worker, kwargs))
            
            # 5. Collect results and update cache
            for future in concurrent.futures.as_completed(tasks_to_run):
                try:
                    result = future.result()
                    chunk_id = result[-1]
                    chunk_cache[chunk_id] = result
                except Exception as e:
                    print(f"Parallel process failed: {e}")

            # 6. Update state and save
            current_chunk_priors = new_chunk_priors
            self._consolidate_and_save(signals, chunk_cache)

        executor.shutdown()

if __name__ == "__main__":
    rng = np.random.default_rng()
    t = np.linspace(0, 1_000, 30_000)
    e = rng.uniform(low=-1, high=1, size=30_000)
    d = (2 * np.sin(2 * np.pi * 4 * t) * np.e ** (-0.010 * t) + 
         3 * np.sin(2 * np.pi * 5 * t) * np.e ** (-0.005 * t) + 
         4 * np.sin(2 * np.pi * 8.95 * t) * np.e ** (-0.020 * t) +
         3 * np.sin(2 * np.pi * 9.05 * t) * np.e ** (-0.025 * t) + e)

    model = BATS(t, d, min_f=0.0, max_f=10.0, mode="global", min_signals=1, 
                 max_signals=5, burn_in=10, typical_set=20, boundaries=True, 
                 std=5, processes=2)
    model.launch()