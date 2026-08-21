import numpy as np
from scipy import signal, stats
import pandas as pd

from obspy.core import UTCDateTime
from obspy.clients.fdsn import Client 

import jax.numpy as jnp
import jax.scipy.special as jsp
import jax

jax.config.update("jax_enable_x64", True)

import math
import random
from tqdm import tqdm

import matplotlib.pyplot as plt



# Bayesian Approach to Signals (BATS)
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

        # If the maximum of the data is less than e, then scale to e. This is required for numerical stability of the ln of the model variances.
        if max(d) < 2.72:
            self.scale = 2.72 / max(d)
            self.d *= self.scale

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
            self.guess_decay_rates = np.maximum(0, self.guess_decay_rates)
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
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 4)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d, n=nfft) * delta

        # Mask the FFT to the specified frequency range
        mask = np.ones_like(frequencies, dtype=bool)

        if minimum_frequency is not None:
            mask &= (frequencies >= minimum_frequency) 

        if maximum_frequency is not None:
            mask &= (frequencies <= maximum_frequency)

        # Return the frequencies and powers with the mask
        return frequencies[mask], power[mask]
    

    def get_peaks(self, x, y):
        # Find the differences in amplitudes
        deltas = np.diff(y)
        peak_indices = []

        # Find all data points where there is a maximum
        N = len(deltas) - 1
        for ind in range(N):
            if deltas[ind] > 0 and deltas[ind + 1] < 0:
                peak_indices.append(ind + 1)

        peak_x = np.array([x[ind] for ind in peak_indices])
        peak_y = np.array([y[ind] for ind in peak_indices])

        # Sort the peaks by largest amplitude
        sort_indices = np.argsort(peak_y)[::-1]

        peak_x = peak_x[sort_indices]
        peak_y = peak_y[sort_indices]

        # Return the x and y values of each peak
        return peak_x, peak_y
    

    def get_decay_rates(self, t, d, frequencies):
        decay_rates = np.zeros(len(frequencies))

        for ind, frequency in enumerate(frequencies):
            # Make a bandwidth at 99.5% - 100.5% of the selected frequency
            minimum_frequency = frequency * 0.995
            maximum_frequency = frequency * 1.005

            # Calculate actual delta from the file data
            delta = np.mean(np.diff(t)) 
            fs = 1.0 / delta

            nyquist = 0.5 * fs
            low = minimum_frequency / nyquist
            high = maximum_frequency / nyquist

            # Filter the data to the chosen frequency band
            order = 4
            b, a = signal.butter(order, [low, high], btype='band')
            detrended = signal.detrend(d)

            frequency_d = signal.filtfilt(b, a, detrended)

            peak_t, peak_d = self.get_peaks(t, frequency_d)

            res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

            # Find the decay rate with linear regression in semilog space
            decay_rates[ind] = -1 * res.slope

        # Return the decay rates
        return decay_rates.tolist()


    def get_model_log_probability(self, model_frequencies, model_decay_rates):
        # Convert frequency to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)

        # Initialize time t and data d
        t = self.t
        d = self.d

        # Find the number of functions r, model rank m, and total number of data points N
        r = len(omegas)
        m = 2 * r
        N = len(d)

        # Construct the sinusoid input and decay rates
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        # Construct the cosine and sine components
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        # Create the model function vector G and Gram matrix g (Bretthorst Page 32)
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

        # Find the orthonormal functions H (Bretthorst Eq. 3.6)
        H = scaled_eigenvectors.T @ G

        # Find the orthonormal function amplitudes h (Bretthorst Eq. 3.13)
        h = H @ d

        # Find the mean-square of the data (Bretthorst Page 17) and mean of the square projection (Bretthorst Eq. 3.15)
        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        # Find probability (Bretthorst Eq. 3.17).
        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * jnp.log10(1 - ratio)

        return log_probability
    

    def get_log_global_likelihood(self, model_frequencies, model_decay_rates):
        # Convert frequency to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)

        # Initialize time t and data d
        t = self.t
        d = self.d

        # Find the number of functions r, model rank m, and total number of data points N
        r = len(omegas)
        m = 2 * r
        N = len(d)

        # Construct the sinusoid input and decay rates
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        # Construct the cosine and sine components
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        # Create the model function vector G and Gram matrix g (Bretthorst Page 32)
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

        # Find the orthonormal functions H (Bretthorst Eq. 3.6)
        H = scaled_eigenvectors.T @ G

        # Find the orthonormal function amplitudes h (Bretthorst Eq. 3.13)
        h = H @ d

        # Find the mean-square data (Bretthorst Page 17) and mean-square projection (Bretthorst Eq. 3.15)
        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        # Define the ratios for integral limits of the three model variances 
        R_delta = jnp.max(jnp.abs(d))                               # Amplitude variance scale
        R_sigma = jnp.max(jnp.abs(d))                               # Noise variance scale
        R_gamma = (0.5 / jnp.mean(jnp.diff(t))) * (t[-1] - t[0])    # Band width variance scale

        q = jnp.concatenate([jnp.array(model_frequencies), jnp.array(model_decay_rates)])

        # Create wrapper function to take an input and return the model's mean-square of the projection
        def get_mean_square_projection(q):
            omegas = q[:r] * 2 * jnp.pi
            alphas = q[r:]

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

            mean_square_projection = (1 / m) * jnp.sum(h ** 2)

            return mean_square_projection

        # Use the wrapper function to get the hessian        
        get_mean_square_projection_hessian = jax.jit(jax.hessian(get_mean_square_projection))

        # Find matrix b (Bretthorst Eq. 4.11)
        b = (-m / 2) * get_mean_square_projection_hessian(q)

        eigenvalues, eigenvectors = jnp.linalg.eigh(b)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        # Compute the global likelihood in log space (Bretthorst Eq. 5.9). The band width term has been modified to account for small parameter values.
        factor = ((m / 2) * jnp.log(2 * jnp.pi)) - (0.5 * jnp.sum(jnp.log(eigenvalues))) - m * jnp.log(R_gamma)
        delta_term = (jsp.gammaln(m / 2))           - (jnp.log(2 * jnp.log(R_delta))) - (m / 2)           * jnp.log((m * mean_square_projection) / 2)
        sigma_term = (jsp.gammaln((N - m - r) / 2)) - (jnp.log(2 * jnp.log(R_sigma))) - ((N - m - r) / 2) * jnp.log((N * mean_square_data - m * mean_square_projection) / 2)
        gamma_term = - (2 * r) * jnp.log(R_gamma)

        log_global_likelihood = delta_term + sigma_term + gamma_term + factor

        return log_global_likelihood
    

    def get_model_statistics(self, model_frequencies, model_decay_rates):
        # Convert frequency to angular frequency
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)

        # Initialize time t and data d
        t = self.t
        d = self.d / self.scale

        # Find the number of functions r, model rank m, and total number of data points N
        r = len(omegas)
        m = 2 * r
        N = len(d)

        # Construct the sinusoid input and decay rates
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        # Construct the cosine and sine components
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        # Create the model function vector G and Gram matrix g (Bretthorst Page 32)
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]

        # Find the orthonormal functions H (Bretthorst Eq. 3.6)
        H = scaled_eigenvectors.T @ G

        # Find the orthonormal function amplitudes h (Bretthorst Eq. 3.13)
        h = H @ d

        # Find the mean-square of the data (Bretthorst Page 17) and mean of the square projection (Bretthorst Eq. 3.15)
        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        # Find probability (Bretthorst Eq. 3.17).
        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * jnp.log10(1 - ratio)

        # Find the estimated noise variance (Bretthorst Eq. 4.7) and SNR (Bretthorst Eq. 4.8)
        estimated_noise_variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))
        SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)

        # Create wrapper function for finding the log probability from the position vector
        def log_probability_wrapper(q):
            model_frequencies = q[:r]
            model_decay_rates = q[r:]

            log_probability = self.get_model_log_probability(model_frequencies, model_decay_rates)

            return log_probability
        
        # Initialize jax functions for the Hessian matrix of the probability distribution
        get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))

        # Find the b matrix (Bretthorst Eq. 4.11)
        b = (-m / 2) * get_log_probability_hessian(np.concatenate([model_frequencies, model_decay_rates]))

        eigenvalues, eigenvectors = jnp.linalg.eigh(b)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        # Find the parameter uncertainties (Bretthorst Eq. 4.13)
        parameter_uncertainties = jnp.sqrt(estimated_noise_variance * jnp.sum((eigenvectors**2) / eigenvalues, axis=1))

        return log_probability, SNR, estimated_noise_variance, parameter_uncertainties
    

    def hamiltonian_monte_carlo(self, 
            guess_frequencies,  # Initial frequencies
            guess_decay_rates,  # Initial decay rates
            B=500,              # Number of burn-in samples
            M=1_000,            # Number of typical set samples
            L=50,               # Number of steps per sample
            epsilon=0.01):      # Stepsize of steps L
        
        # Combine the frequencies and decay rates into a position vector
        current_q = np.concatenate([guess_frequencies, guess_decay_rates])

        # Specify the dimension of the parameter space
        dimension = len(current_q)

        # Find the number of functions
        r = int(len(current_q) / 2)
        
        # Create wrapper function for finding the log probability from the position vector
        def log_probability_wrapper(q):
            model_frequencies = q[:r]
            model_decay_rates = q[r:]

            log_probability = self.get_model_log_probability(model_frequencies, model_decay_rates)

            return log_probability
        
        # Initialize jax functions for the gradient and Hessian matrix of the probability distribution
        get_log_probability_gradient = jax.jit(jax.value_and_grad(log_probability_wrapper))
        get_log_probability_hessian = jax.jit(jax.hessian(log_probability_wrapper))

        # Get the diagonal of the Hessian matrix
        hessian_diagonal = -np.diag(get_log_probability_hessian(current_q))
        
        # Use the Hessian diagonal matrix to inform the mass matrix
        mass_matrix = jnp.abs(jnp.array(hessian_diagonal))

        # Get the initial log probability and log gradient
        current_log_probability, current_log_probability_gradient = get_log_probability_gradient(current_q)
        
        # Initialize the history and acceptances of the burn-in sampling phase
        burn_in_history = np.zeros(B, dtype=object)
        acceptances = 0

        # Run the burn-in sampling loop for B samples
        for iteration in tqdm(range(B), desc="Burn-in Samples"):
            # Pick a random sample of the dimension of the parameter space for the momentum vector informed by the mass matrix
            p = np.random.normal(0, np.sqrt(mass_matrix), dimension)

            # Find the kinetic energy of the particle current position
            current_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            # Initialize the position vector for the Leapfrog integrator
            q = current_q.copy()
            
            # Run the Leapfrog integrator for L samples
            for _ in range(L):
                # Update the momentum (1/2)
                p = p + (epsilon / 2.0) * current_log_probability_gradient

                # Update the position one full-step
                q = q + epsilon * (p / mass_matrix)

                # Find the the new log gradient
                _, proposal_log_probability_gradient = get_log_probability_gradient(q)
                proposal_log_probability_gradient = np.array(proposal_log_probability_gradient)

                # Update the momentum (2/2)
                p = p + (epsilon / 2.0) * proposal_log_probability_gradient

            # Get the log probability of the proposal distribution
            proposal_log_probability = self.get_model_log_probability(q[:r], q[r:])

            # Find the kinetic energy of the particle proposal position
            proposal_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            # Calculate the current Hamiltonian and proposal Hamiltonian
            current_H = -current_log_probability + current_T
            proposal_H = -proposal_log_probability + proposal_T

            # Find the Metropolis-Hastings criterion and randomly draw a number from 0 to 1
            a = current_H - proposal_H              # Hamiltonians are subtracted since they are in log space
            c = np.log(random.random() + 1e-100)

            # If the randomly drawn number is less than the criterion, then accept the new position
            if c < a:
                current_q = q
                current_q = np.maximum(0, current_q)
                current_log_probability = proposal_log_probability
                current_log_probability_gradient = proposal_log_probability_gradient

                acceptances += 1

            # Record the position and probability of the particle for each iteration
            burn_in_history[iteration] = [q[:r], q[r:], current_log_probability]


        print(f"Burn-in acceptance rate: {round(100 * acceptances / B, 4)}%")
        burn_in_acceptance_rate = round(100 * acceptances / B, 4)

        typical_set_history = np.zeros(M, dtype=object)
        acceptances = 0

        # Run the typical set sampling loop for M samples
        for iteration in tqdm(range(M), desc="Typical Set Samples"):
            # Pick a random sample of the dimension of the parameter space for the momentum vector informed by the mass matrix
            p = np.random.normal(0, np.sqrt(mass_matrix), dimension)

            # Find the kinetic energy of the particle current position
            current_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            # Initialize the position vector for the Leapfrog integrator
            q = current_q.copy()
            
            # Run the Leapfrog integrator for L samples
            for _ in range(L):
                # Update the momentum (1/2)
                p = p + (epsilon / 2.0) * current_log_probability_gradient

                # Update the position one full-step
                q = q + epsilon * (p / mass_matrix)

                # Find the the new log gradient
                _, proposal_log_probability_gradient = get_log_probability_gradient(q)
                proposal_log_probability_gradient = np.array(proposal_log_probability_gradient)

                # Update the momentum (2/2)
                p = p + (epsilon / 2.0) * proposal_log_probability_gradient

            # Get the log probability of the proposal distribution
            proposal_log_probability = self.get_model_log_probability(q[:r], q[r:])

            # Find the kinetic energy of the particle proposal position
            proposal_T = 0.5 * np.sum(p ** 2 / mass_matrix)

            # Calculate the current Hamiltonian and proposal Hamiltonian
            current_H = -current_log_probability + current_T
            proposal_H = -proposal_log_probability + proposal_T

            # Find the Metropolis-Hastings criterion and randomly draw a number from 0 to 1
            a = current_H - proposal_H              # Hamiltonians are subtracted since they are in log space
            c = np.log(random.random() + 1e-100)

            # If the randomly drawn number is less than the criterion, then accept the new position
            if c < a:
                current_q = q
                current_log_probability = proposal_log_probability
                current_log_probability_gradient = proposal_log_probability_gradient

                acceptances += 1

            # Record the position and probability of the particle for each iteration
            typical_set_history[iteration] = [q[:r], q[r:], current_log_probability]

        print(f"Typical set acceptance rate: {round(100 * acceptances / M, 4)}%")
        typical_set_acceptance_rate = round(100 * acceptances / M, 4)

        return burn_in_history, typical_set_history, burn_in_acceptance_rate, typical_set_acceptance_rate


    def run(self,
            functions=None,                     # If functions is specified, then do not iterate multiple models
            lower_bound_model_functions=1,      # Otherwise, iterate from a lower bound to upper bound of number of model functions to estimate
            upper_bound_model_functions=5,
            B=1_000,                            # B: number of burn-in samples
            M=2_000,                            # M: number of typical set samples
            L=50,                               # L: number of Leapfrog integrations per chain
            epsilon=0.001):                     # epsilon: step size for Leapfrog integrations

        if functions is None:
            # Specify the number of models needed
            models = upper_bound_model_functions - lower_bound_model_functions + 1

            # Initialize the history for each model
            histories = np.zeros(models, dtype=object)

            burn_in_acceptance_rates = np.zeros(models)
            typical_set_acceptance_rates = np.zeros(models)

            # Iterate through each model
            for additional_functions in range(models):
                print(f"\nNow creating a {lower_bound_model_functions + additional_functions}-function model!")

                # Retrieve the number of frequencies and decay rates to use
                self.signals = (lower_bound_model_functions + additional_functions)
                guess_frequencies = self.guess_frequencies[:(lower_bound_model_functions + additional_functions)]
                guess_decay_rates = self.guess_decay_rates[:(lower_bound_model_functions + additional_functions)]

                _, history, burn_in_acceptance_rate, typical_set_acceptance_rate = self.hamiltonian_monte_carlo(guess_frequencies, guess_decay_rates, B=B, M=M, L=L, epsilon=epsilon)

                # Save the model's history
                histories[additional_functions] = history
                burn_in_acceptance_rates[additional_functions] = burn_in_acceptance_rate
                typical_set_acceptance_rates[additional_functions] = typical_set_acceptance_rate

            # Initialize the global likelihoods, SNRs, unceratinties, and variances
            global_likelihoods = np.zeros(models)
            SNRs = np.zeros(models)
            variances = np.zeros(models)
            model_parameter_uncertainties = np.zeros(models, dtype=object)

            with open("BATS_model_results.txt", "w", encoding="utf-8") as f:
                # Iterate through each history
                for ind, history in enumerate(histories):
                    # Parse the highest probability frequencies and decay rates for each model
                    sorted_history = sorted(history, key=lambda x: x[2], reverse=True)

                    realized_frequencies = sorted_history[0][0]
                    realized_decay_rates = sorted_history[0][1]

                    # Find the global likelihood, SNRs, and variances for the highest probability parameters
                    _, SNR, var, parameter_uncertainties = self.get_model_statistics(realized_frequencies, realized_decay_rates)
                    SNRs[ind] = SNR
                    variances[ind] = var
                    model_parameter_uncertainties[ind] = parameter_uncertainties
                    
                    global_likelihood = self.get_log_global_likelihood(realized_frequencies, realized_decay_rates)
                    global_likelihoods[ind] = global_likelihood

                    r = len(realized_frequencies)

                    freq_uncert = parameter_uncertainties[:r]
                    decay_uncert = parameter_uncertainties[r:]

                    print(f"\nDetails for {ind + lower_bound_model_functions}-frequency model:", file=f)
                    print(f"{'Signal':<8} | {'Frequency ± Uncertainty':<28} | {'Decay Rate ± Uncertainty':<32}", file=f)
                    print("-" * 90, file=f)
                    
                    for i in range(r):
                        f_val = realized_frequencies[i]
                        f_err = freq_uncert[i]
                        d_val = realized_decay_rates[i]
                        d_err = decay_uncert[i]
                        
                        print(f"{i+1:<8} | {f_val:>14.10f} ± {f_err:<14.10f} | {d_val:>16.12f} ± {d_err:<16.12f}", file=f)
                    print("-" * 90, file=f)

                # Find the highest probability model and print the results for each model
                max_log_l = np.nanmax(global_likelihoods)
                exp_terms = np.exp(global_likelihoods - max_log_l)
                probs = exp_terms / np.nansum(exp_terms)
                best_index = np.nanargmax(global_likelihoods)
                
                print(f"{'\nOrder (r)':<10} | {'Log-Likelihood':<18} | {'SNRs':<12} | {'Variance':<12} | {'Probability Weight':<12} | {'B %':<12} | {'M %':<12}", file=f)
                print("-" * 90, file=f)
                for i, log_l in enumerate(global_likelihoods):
                    print(f"{i + lower_bound_model_functions:<10} | {log_l:<18.2f} | {SNRs[i]:<12.2f} | {np.log(variances[i]):<12.2f} | {probs[i]:<12.4e} | {burn_in_acceptance_rates[i]:<12.4f} | {typical_set_acceptance_rates[i]:<12.4f}", file=f)
                print("-" * 90, file=f)
                print(f"The best model is the {best_index + lower_bound_model_functions}-frequency model!", file=f)

        else:
            # TODO: make one frequency sampling work
            history = self.hamiltonian_monte_carlo(self.guess_frequencies, self.guess_decay_rates)



if __name__ == "__main__":
    def get_synthetic_data(filename, minimum_frequency=None, maximum_frequency=None):
        data = pd.read_csv(filename, sep=' ', header=None, encoding='ascii')

        t = data.iloc[:, 0].values
        d = data.iloc[:, 1].values
        
        # Calculate actual delta from the file data
        delta = np.mean(np.diff(t)) 
        fs = 1.0 / delta

        nyquist = 0.5 * fs
        low = minimum_frequency / nyquist
        high = maximum_frequency / nyquist

        # Filter the data to the chosen frequency band
        order = 4
        b, a = signal.butter(order, [low, high], btype='band')
        detrended = signal.detrend(d)

        d = signal.filtfilt(b, a, detrended)

        # Discard the first 10 hours of data for sampling rate of 0.004 Hz
        t = t[144:]
        d = d[144:]

        return t, d


    def get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency):
        client = Client('IRIS')

        inventory = client.get_stations(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time, level='response')
        stream = client.get_waveforms(network=network, station=station, location=location, channel=channel, starttime=start_time, endtime=end_time)

        trace = stream[stream_index]
        trace.detrend('constant')
        trace.remove_response(inventory=inventory, output="ACC")

        if minimum_frequency and maximum_frequency:
            trace.filter('bandpass', freqmin=minimum_frequency, freqmax=maximum_frequency)
            
        # Data is by default 1 Hz, so decimate to 0.004 Hz to match synthetic data
        trace.decimate(5, no_filter=False)
        trace.decimate(5, no_filter=False)
        trace.decimate(5, no_filter=False)
        trace.decimate(2, no_filter=False)

        delta = trace.stats.delta 
        N = len(trace)
        t = np.arange(N) * delta
        
        d = np.array(trace.data)

        # Discard the first 10 hours of data for sampling rate of 0.004 Hz
        t = t[144:]
        d = d[144:]

        return t, d




# -------------------------------------------------------------------------------

    minimum_frequency = 0.0005       # Minimum frequency (Hz)
    maximum_frequency = 0.0012       # Maximum frequency (Hz)

    network = "IU"                   # Network
    station = "KIP"                  # Station
    channel = "LHZ"                  # Channel
    location = "00"                  # Location

    stream_index = 0                                    # Stream index
    start_time = UTCDateTime('2025-07-31T06:24:50')     # Start time
    end_time = UTCDateTime('2025-08-11T05:24:50')       # End time

    # File path may need to be changed depending on root directory
    file_path = f"timeseries_Russia/{network}_{station}_TS.ascii" 

    lower_bound_model_functions = 15
    upper_bound_model_functions = 17

    B = 40              # Number of burn-in samples
    M = 40              # Number of typical set samples
    L = 30                 # Number of Leapfrog integrations per sample
    epsilon = 0.003        # Step size of Leapfrog integrations
    
    data = "synthetic"

# -------------------------------------------------------------------------------




    if data == "observed":
        t, d = get_observed_data(network, station, channel, location, stream_index, start_time, end_time, minimum_frequency, maximum_frequency)
    elif data == "synthetic":
        t, d = get_synthetic_data(file_path, minimum_frequency, maximum_frequency)

    model = BATS(t, d, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
    model.run(lower_bound_model_functions=lower_bound_model_functions, upper_bound_model_functions=upper_bound_model_functions, B=B, M=M, L=L, epsilon=epsilon)
