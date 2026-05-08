import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks
from scipy.optimize import minimize
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import random
import math
from tqdm import tqdm
import matplotlib.animation as animation
import matplotlib.pyplot as plt


class BATS():
    def __init__(self, 
                 t,                         # Time (s) of the data
                 d,                         # Measurement for each time
                 minimum_frequency=0,       # Minimum (Hz) frequency to sample
                 maximum_frequency=None,    # Maximum (Hz) frequency to sample
                 signals=None,              # Number of sinusoids expected in the data
                 guess_frequencies=None,    # Guess frequencies (len(guess_frequencies) == signals)
                 guess_decay_rates=None,    # Guess decay rates (decay_rate must be specified)
                 decay_type="lorentzian"):  # Decay type: None, Lorenztian, Gaussian
        
        # Set the time t and data d
        self.t = t
        self.d = d

        # Set the minimum frequency
        self.minimum_frequency = minimum_frequency

        if maximum_frequency is None:
            # Default the maximum frequency to the Nyquist frequency
            self.maximum_frequency = 0.5 / np.mean(np.diff(t))
        else:
            self.maximum_frequency = maximum_frequency

        if guess_frequencies is None:
            # Use the FFT to extract the initial frequencies TODO: add Bretthorst 1 frequency residual model to get guess frequencies
            f, p = self.get_fft(t, d, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
            self.guess_frequencies, _ = self.get_peaks(f, abs(p))
            self.guess_frequencies = self.guess_frequencies.tolist()
        else:
            self.guess_frequencies = guess_frequencies.tolist()

        if guess_decay_rates is None:
            # Calculate the decay rates from banding the guess frequencies individually in d
            self.guess_decay_rates = self.get_decay_rates(t, d, self.guess_frequencies)
        else:
            self.guess_decay_rates = guess_decay_rates.tolist()

        if signals is None:
            # Set signals to the total found
            self.signals = len(self.guess_frequencies)
        else:
            # Limit the number of signals if specified
            self.signals = signals

            self.guess_frequencies = self.guess_frequencies[:signals]
            self.guess_decay_rates = self.guess_decay_rates[:signals]

        # Set the decay type TODO: add functionality for Gaussian and None decay
        self.decay_type = decay_type


    def get_fft(self, t, d, minimum_frequency=None, maximum_frequency=None):
        # Find the sample rate
        delta = np.mean(np.diff(t)) 

        # Apply a window function
        taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
        d = d * taper
        
        # Find the FFT
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 2)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d, n=nfft) * delta

        mask = np.ones_like(frequencies, dtype=bool)

        if minimum_frequency is not None:
            mask &= (frequencies >= minimum_frequency) 

        if maximum_frequency is not None:
            mask &= (frequencies <= maximum_frequency)

        # Return the frequencies and powers masked
        return frequencies[mask], power[mask]
    

    def get_peaks(self, x, y):
        deltas = np.diff(y)
        peak_indices = []

        N = len(deltas) - 1
        for ind in range(N):
            if deltas[ind] > 0 and deltas[ind + 1] < 0:
                peak_indices.append(ind + 1)

        peak_x = np.array([x[ind] for ind in peak_indices])
        peak_y = np.array([y[ind] for ind in peak_indices])

        sort_indices = np.argsort(peak_y)[::-1]

        peak_x = peak_x[sort_indices]
        peak_y = peak_y[sort_indices]

        return peak_x, peak_y
    

    def get_decay_rates(self, t, d, frequencies):
        decay_rates = np.zeros(len(frequencies))

        for ind, frequency in enumerate(frequencies):
            minimum_frequency = frequency * 0.99
            maximum_frequency = frequency * 1.01

            # Calculate actual delta from the file data
            delta = np.mean(np.diff(t)) 
            fs = 1.0 / delta

            nyquist = 0.5 * fs
            low = minimum_frequency / nyquist
            high = maximum_frequency / nyquist

            order = 4
            b, a = signal.butter(order, [low, high], btype='band')
            detrended = signal.detrend(d)

            frequency_d = signal.filtfilt(b, a, detrended)

            peak_t, peak_d = self.get_peaks(t, frequency_d)

            res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

            decay_rates[ind] = -1 * res.slope

        return decay_rates.tolist()


    def get_model_probability(self, model_frequencies, model_decay_rates):
        # Convert frequencies to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)
        
        t = self.t
        d = self.d

        r = len(omegas)
        m = 2 * r
        N = len(d)
        
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]
        H = scaled_eigenvectors.T @ G

        h = H @ d

        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * jnp.log(jnp.maximum(1 - ratio, 1e-15))

        return log_probability
    

    def get_global_likelihood(self, model_frequencies, model_decay_rates):
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)
        
        t = self.t
        d = self.d

        r = len(omegas)
        m = 2 * r
        N = len(d)
        
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        
        mask = eigenvalues > 1e-10
        safe_eigenvalues = jnp.where(mask, eigenvalues, 1.0)
        scaled_eigenvectors = eigenvectors * jnp.where(mask, 1.0 / jnp.sqrt(safe_eigenvalues), 0.0)[None, :]
        
        H = scaled_eigenvectors.T @ G
        h = H @ d

        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        R_sigma, R_delta = 1.67e7, 1.67e7
        R_gamma = (0.5 / jnp.mean(jnp.diff(t))) * (t[-1] - t[0])

        term_delta = jsp.gammaln(m / 2) - m * jnp.log(R_delta) - (m / 2) * jnp.log(jnp.maximum(m * mean_square_projection / 2, 1e-15))
        term_gamma = - (r * jnp.log(R_gamma))
        term_sigma = jsp.gammaln((N - m - r) / 2) - jnp.log(jnp.log10(R_sigma)) - ((N - m - r) / 2) * jnp.log(jnp.maximum((N * mean_square_data - m * mean_square_projection) / 2, 1e-15))
        
        def ms_projection_wrapper(q):
            f, a = q[:r], q[r:]
            omega_l = f * 2 * jnp.pi
            decay_l = jnp.exp(-a[:, None] * t[None, :])
            G_l = jnp.vstack((jnp.cos(omega_l[:, None] * t[None, :]) * decay_l, 
                              jnp.sin(omega_l[:, None] * t[None, :]) * decay_l))
            vals, vecs = jnp.linalg.eigh(G_l @ G_l.T)
            H_l = (vecs / jnp.sqrt(jnp.maximum(vals, 1e-4))[None, :]).T @ G_l
            return jnp.sum((H_l @ d) ** 2) / m

        q_combined = jnp.concatenate([jnp.array(model_frequencies), jnp.array(model_decay_rates)])
        b_matrix = (-m / 2) * jax.hessian(ms_projection_wrapper)(q_combined)
        
        sign, logdet = jnp.linalg.slogdet(b_matrix + jnp.eye(2 * r) * 1e-7)
        logdet = jnp.where((sign > 0) & jnp.isfinite(logdet), logdet, 150.0)
        laplace_factor = (r * jnp.log(2 * jnp.pi)) - (0.5 * logdet)

        global_likelihood = term_delta + term_gamma + term_sigma + laplace_factor

        return global_likelihood
    

    def get_model_statistics(self, model_frequencies, model_decay_rates):
        # Convert frequencies to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)
        
        t = self.t
        d = self.d

        r = len(omegas)
        m = 2 * r
        N = len(d)
        
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]
        H = scaled_eigenvectors.T @ G

        h = H @ d

        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * jnp.log(jnp.maximum(1 - ratio, 1e-15))

        estimated_noise_variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))
        SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)

        return log_probability, SNR, estimated_noise_variance


    def hamiltonian_monte_carlo(self, 
            guess_frequencies,
            guess_decay_rates,
            B=500,              # Number of burn-in samples
            M=1_000,            # Number of typical set samples
            L=50,               # Number of steps per sample
            epsilon=0.01):      # Stepsize of steps L

        current_q = np.concatenate([guess_frequencies, guess_decay_rates])
        parameters = len(current_q)
        functions = int(self.signals)

        def probability_wrapper(q):
            frequencies = q[:functions]
            decay_rates = q[functions:]

            log_probability = self.get_model_probability(frequencies, decay_rates)

            return log_probability

        get_log_probability_gradient = jax.jit(jax.value_and_grad(probability_wrapper))
        get_log_probability_hessian = jax.jit(jax.hessian(probability_wrapper))

        hessian = get_log_probability_hessian(current_q)
        hessian_diag = -np.diag(hessian)
        hessian_diag = np.where((hessian_diag > 0) & np.isfinite(hessian_diag), hessian_diag, 1e-6)
        
        mass_matrix = np.array(hessian_diag)

        current_log_probability, log_probability_gradient = get_log_probability_gradient(current_q)
        log_probability_gradient = np.array(log_probability_gradient)

        burn_in_history = np.zeros(B, dtype=object)
        acceptances = 0

        for iteration in tqdm(range(B), desc="Burn-in Samples"):
            p = np.random.normal(0, np.sqrt(mass_matrix), parameters)
            current_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            q = current_q.copy()
            current_log_probability_gradient = log_probability_gradient.copy()
            
            for _ in range(L):
                p = p + (epsilon / 2.0) * current_log_probability_gradient

                q = q + epsilon * (p / mass_matrix)

                _, proposal_log_probability_gradient = get_log_probability_gradient(q)
                proposal_log_probability_gradient = np.array(proposal_log_probability_gradient)

                p = p + (epsilon / 2.0) * proposal_log_probability_gradient

            proposal_log_probability = self.get_model_probability(q[:functions], q[functions:])
            proposal_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            current_H = -current_log_probability + current_T
            proposal_H = -proposal_log_probability + proposal_T

            a = current_H - proposal_H
            c = np.log(random.random() + 1e-100)

            if c < a:
                current_q = q
                current_log_probability = proposal_log_probability
                log_probability_gradient = proposal_log_probability_gradient

                acceptances += 1

            burn_in_history[iteration] = [q[:functions], q[functions:], current_log_probability]

            if iteration > B // 2:
                available_burn_in_history = burn_in_history[:iteration]
                frequencies, decay_rates, _ = zip(*available_burn_in_history)
                samples = np.hstack([np.array(frequencies), np.array(decay_rates)])
                scales = np.std(samples, axis=0)
                scales = np.maximum(scales, np.abs(current_q) * 0.01)
                
                mass_matrix = 1.0 / (scales ** 2)

        print(f"Burn-in acceptance rate: {round(100 * acceptances / B, 4)}%")

        typical_set_history = np.zeros(M, dtype=object)
        acceptances = 0

        for iteration in tqdm(range(M), desc="Typical Set Samples"):
            p = np.random.normal(0, np.sqrt(mass_matrix), parameters)
            current_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            q = current_q.copy()
            current_log_probability_gradient = log_probability_gradient.copy()
            
            for _ in range(L):
                p = p + (epsilon / 2.0) * current_log_probability_gradient

                q = q + epsilon * (p / mass_matrix)

                _, proposal_log_probability_gradient = get_log_probability_gradient(q)
                proposal_log_probability_gradient = np.array(proposal_log_probability_gradient)

                p = p + (epsilon / 2.0) * proposal_log_probability_gradient

            proposal_log_probability = self.get_model_probability(q[:functions], q[functions:])
            proposal_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            current_H = -current_log_probability + current_T
            proposal_H = -proposal_log_probability + proposal_T

            a = current_H - proposal_H
            c = np.log(random.random() + 1e-100)

            if c < a:
                current_q = q
                current_log_probability = proposal_log_probability
                log_probability_gradient = proposal_log_probability_gradient

                acceptances += 1

            typical_set_history[iteration] = [q[:functions], q[functions:], current_log_probability]
        
        print(f"Typical set acceptance rate: {round(100 * acceptances / M, 4)}%")

        return typical_set_history
    

    def metropolis_hastings(self, 
                            guess_frequencies, 
                            guess_decay_rates, 
                            B=500,              # Number of burn-in samples
                            M=1_000,            # Number of typical set samples
                            epsilon=0.01):      # Proposal step scale

        current_q = np.concatenate([guess_frequencies, guess_decay_rates])
        parameters = len(current_q)
        functions = int(self.signals)

        current_log_probability = self.get_model_probability(current_q[:functions], current_q[functions:])

        burn_in_history = np.zeros(B, dtype=object)
        acceptances = 0

        for iteration in tqdm(range(B), desc="Burn-in Samples"):
            # Random Walk Proposal: q_new = q_old + N(0, epsilon * scales)
            q = current_q + np.random.normal(0, epsilon, parameters)

            proposal_log_probability = self.get_model_probability(q[:functions], q[functions:])

            # Metropolis Acceptance Ratio (Log Space)
            a = proposal_log_probability - current_log_probability
            c = np.log(random.random() + 1e-100)

            if c < a:
                current_q = q
                current_log_probability = proposal_log_probability
                acceptances += 1

            # Store the current state (after potential update)
            burn_in_history[iteration] = [current_q[:functions], current_q[functions:], current_log_probability]


        print(f"Burn-in acceptance rate: {round(100 * acceptances / B, 4)}%")

        typical_set_history = np.zeros(M, dtype=object)
        acceptances = 0

        for iteration in tqdm(range(M), desc="Typical Set Samples"):
            # Random Walk Proposal
            q = current_q + np.random.normal(0, epsilon, parameters)

            proposal_log_probability = self.get_model_probability(q[:functions], q[functions:])

            a = proposal_log_probability - current_log_probability
            c = np.log(random.random() + 1e-100)

            if c < a:
                current_q = q
                current_log_probability = proposal_log_probability
                acceptances += 1

            typical_set_history[iteration] = [current_q[:functions], current_q[functions:], current_log_probability]
        
        print(f"Typical set acceptance rate: {round(100 * acceptances / M, 4)}%")

        return typical_set_history
    

    def run(self,
            functions=None,                     # If functions is specified, then do not iterate multiple models
            lower_bound_model_functions=1,      # Otherwise, iterate from a lower bound to upper bound of number of model functions to estimate
            upper_bound_model_functions=5):

        if functions is None:
            models = upper_bound_model_functions - lower_bound_model_functions + 1

            histories = np.zeros(models, dtype=object)

            for additional_functions in range(models):
                print(f"\nNow creating a {lower_bound_model_functions + additional_functions}-function model!")

                self.signals = (lower_bound_model_functions + additional_functions)
                guess_frequencies = self.guess_frequencies[:(lower_bound_model_functions + additional_functions)]
                guess_decay_rates = self.guess_decay_rates[:(lower_bound_model_functions + additional_functions)]

                history = self.hamiltonian_monte_carlo(guess_frequencies, guess_decay_rates, B=1, M=1, L=1, epsilon=3e-9)
                histories[additional_functions] = history

            global_likelihoods = np.zeros(models)
            SNRs = np.zeros(models)
            variances = np.zeros(models)

            for ind, history in enumerate(histories):
                sorted_history = sorted(history, key=lambda x: x[2], reverse=True)

                realized_frequencies = sorted_history[0][0]
                realized_decay_rates = sorted_history[0][1]

                prob, SNR, var = self.get_model_statistics(realized_frequencies, realized_decay_rates)
                SNRs[ind] = SNR
                variances[ind] = var

                global_likelihood = self.get_global_likelihood(realized_frequencies, realized_decay_rates)
                global_likelihoods[ind] = global_likelihood

            max_log_l = np.nanmax(global_likelihoods)
            exp_terms = np.exp(global_likelihoods - max_log_l)
            probs = exp_terms / np.nansum(exp_terms)
            best_index = np.nanargmax(global_likelihoods)

            print(f"{'Order (r)':<10} | {'Log-Likelihood':<18} | {'SNRs':<12} | {'Variance':<12} | {'Prob Weight':<12}")
            print("-" * 77)
            for i, log_l in enumerate(global_likelihoods):
                print(f"{i + lower_bound_model_functions:<10} | {log_l:<18.2f} | {SNRs[i]:<12.2f} | {np.log(variances[i]):<12.2f} | {probs[i]:<12.4e}")
            print("-" * 77)
            print(f"The best model is the {best_index + lower_bound_model_functions}-frequency model!")

        else:
            history = self.hamiltonian_monte_carlo(self.guess_frequencies, self.guess_decay_rates)

        