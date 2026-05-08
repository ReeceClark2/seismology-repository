import math
import random

import numpy as np
from scipy import signal, stats

import tqdm

import matplotlib.pyplot as plt

# Bayesian Approach to Signals
class BATS(): 
    def __init__(self, t, d, minimum_frequency, maximum_frequency, guess_frequencies=None, guess_decay_rates=None, threshold=50):
        self.t = t
        self.d = d

        self.minimum_frequency = minimum_frequency
        self.maximum_frequency = maximum_frequency
        
        if guess_frequencies is not None:
            self.guess_frequencies = guess_frequencies
        else:
            frequencies, power = self.compute_fft(t, d, minimum_frequency, maximum_frequency)
            self.guess_frequencies, _ = self.find_peaks(frequencies, abs(power), threshold)

        if self.guess_frequencies == 0:
            print("0 frequencies detected! Select a lower tolerance threshold!")
        else:
            print(f"{len(self.guess_frequencies)} frequencies detected.")

        if guess_decay_rates is not None:
            self.guess_decay_rates = guess_decay_rates
        else:
            self.guess_decay_rates = self.get_decay_rates(t, d, self.guess_frequencies)


    def get_decay_rates(self, t, d, frequencies):
        decay_rates = np.zeros(len(frequencies))

        for ind, frequency in enumerate(frequencies):
            minimum_frequency = frequency - 0.00003
            maximum_frequency = frequency + 0.00003

            # Calculate actual delta from the file data
            delta = np.mean(np.diff(t)) 
            fs = 1.0 / delta

            nyquist = 0.5 * fs
            low = minimum_frequency / nyquist
            high = maximum_frequency / nyquist

            order = 4
            b, a = signal.butter(order, [low, high], btype='band')

            # Detrending prevents edge offsets from blowing up the filter
            detrended = signal.detrend(d)

            frequency_d = signal.filtfilt(b, a, detrended)

            peak_t, peak_d = self.find_peaks(t, frequency_d, threshold=0)

            res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

            decay_rates[ind] = -1 * res.slope

        return decay_rates


    def find_peaks(self, x, y, threshold):
        threshold = threshold * np.median(y)
        deltas = np.diff(y)
        peak_indices = []

        N = len(deltas) - 1
        for ind in range(N):
            if deltas[ind] > 0 and deltas[ind + 1] < 0 and y[ind + 1] > threshold:
                peak_indices.append(ind + 1)

        peak_x = []
        peak_y = []
        for ind in peak_indices:
            peak_x.append(x[ind])
            peak_y.append(y[ind])

        return peak_x, peak_y


    def compute_fft(self, t, d, minimum_frequency=None, maximum_frequency=None):
        delta = np.mean(np.diff(t)) 

        taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
        d = d * taper
        
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 4)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d, n=nfft) * delta

        mask = np.ones_like(frequencies, dtype=bool)

        if minimum_frequency is not None:
            mask &= (frequencies >= minimum_frequency) 

        if maximum_frequency is not None:
            mask &= (frequencies <= maximum_frequency)

        return frequencies[mask], power[mask]
    

    def compute_model_probability(self, model_frequencies, model_decay_rates, estimated_noise_variance_flag=False, SNR_flag=False, save_model_flag=False, model_title=None):
        model_frequencies = np.array(model_frequencies) * 2 * np.pi
        
        t = self.t
        d = self.d

        r = len(model_frequencies)
        m = 2 * r
        N = len(d)
        
        # Initialize functions and Gram matrix (Bretthorst Eq. 3.1 & Eq. 3.4).
        G = np.zeros(m, dtype=object)
        g = np.zeros((m, m))

        # Populate model function arrays (Bretthorst page 32).
        for j in range(r):
            G[j] = np.cos(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)
            G[j + r] = np.sin(model_frequencies[j] * t) * np.e ** (-model_decay_rates[j] * t)

        # Populate Gram matrix (Bretthorst Eq. 3.4).
        for j in range(r):
            for k in range(r):
                g[j, k] = np.sum(G[j] * G[k])
                g[j + r, k] = np.sum(G[j + r] * G[k])
                g[j, k + r] = np.sum(G[j] * G[k + r])
                g[j + r, k + r] = np.sum(G[j + r] * G[k + r])

        # Find eigenvalues and eigenvectors (Bretthorst 33).
        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        # Find orthonormal basis functions (Bretthorst Eq. 3.5).
        H = np.zeros(m, dtype=object)
        for j, _ in enumerate(H):
            for k in range(m):
                H[j] += (1 / np.sqrt(eigenvalues[j])) * eigenvectors[k][j] * G[k]

        # Find projections of data onto orthonormal basis functions, orthonormal amplitudes (Bretthorst Eq. 3.13).
        h = np.zeros(m)
        for j, _ in enumerate(h):
            h[j] = np.sum(d * H[j])

        # Calculate dbar (Bretthorst page 17).
        mean_square_data = (1 / N) * np.sum(d ** 2)

        # Calculate hbar (Bretthorst Eq. 3.15).
        mean_square_projection = (1 / m) * np.sum(h ** 2)

        # Find probability (Bretthorst Eq. 3.17).
        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * np.log10(1 - ratio)

        estimated_noise_variance = None
        if estimated_noise_variance_flag == True or SNR_flag == True:
            estimated_noise_variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))

        SNR = None
        if SNR_flag == True:
            SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)

        if save_model_flag == True:
            model = np.zeros(len(t))
            for j, _ in enumerate(h):
                model += h[j] * H[j]

            plt.figure(figsize=(10, 6))

            plt.plot(t, model, color="sandybrown", alpha=0.8, label="Model")
            plt.plot(t, d, color="cornflowerblue", alpha=0.8, label="Data")
            plt.plot(t, d - model, color="red", alpha=0.4, label="Residuals")

            plt.xlim(min(t), max(t))
            plt.xlabel("Time (s)")
            plt.ylabel(r"Acceleration (m/s$^2$)")
            plt.tight_layout()
            plt.legend()

            if model_title == None:
                plt.title("Time Series")
                plt.savefig("BATS_model.png", dpi=300)
            else:
                plt.title(f"Time Series")
                plt.savefig(f"{model_title}.png", dpi=300)

            plt.close('all')
            f, p = self.compute_fft(t, d - model, self.minimum_frequency, self.maximum_frequency)
            f1, p1 = self.compute_fft(t, d, self.minimum_frequency, self.maximum_frequency)
            plt.plot(f, abs(p), color='red')
            plt.plot(f1, abs(p1), color='black')
            plt.xlim(self.minimum_frequency, self.maximum_frequency)
            plt.show()

        return log_probability, mean_square_projection, estimated_noise_variance, SNR


    def compute_model_parameter_uncertainties(self, model_frequencies, model_decay_rates):
        m = len(model_frequencies) + len(model_decay_rates)
        r = int(m / 2)

        original_theta = np.array(list(model_frequencies) + list(model_decay_rates), dtype=float)
        H = np.zeros((m, m))  # Renamed 'b' to 'H' to represent the Hessian

        def get_prob(theta):
            freqs = theta[:r]
            decays = theta[r:]
            _, msp, _, _ = self.compute_model_probability(freqs, decays, estimated_noise_variance_flag=True)
            return msp

        f0 = get_prob(original_theta)
    
        eps = np.finfo(float).eps
        h = np.cbrt(eps) * np.maximum(np.abs(original_theta), 1e-6)
        
        f_plus = np.zeros(m)
        f_minus = np.zeros(m)
        
        for j in range(m):
            theta_plus = original_theta.copy()
            theta_plus[j] += h[j]
            f_plus[j] = get_prob(theta_plus)
            
            theta_minus = original_theta.copy()
            theta_minus[j] -= h[j]
            f_minus[j] = get_prob(theta_minus)
            
            H[j, j] = (f_plus[j] - 2 * f0 + f_minus[j]) / (h[j] ** 2)

        for j in range(m):
            for k in range(j + 1, m):
                theta_pp = original_theta.copy(); theta_pp[j] += h[j]; theta_pp[k] += h[k]
                theta_pm = original_theta.copy(); theta_pm[j] += h[j]; theta_pm[k] -= h[k]
                theta_mp = original_theta.copy(); theta_mp[j] -= h[j]; theta_mp[k] += h[k]
                theta_mm = original_theta.copy(); theta_mm[j] -= h[j]; theta_mm[k] -= h[k]
                
                f_pp = get_prob(theta_pp)
                f_pm = get_prob(theta_pm)
                f_mp = get_prob(theta_mp)
                f_mm = get_prob(theta_mm)
                
                mixed_partial = (f_pp - f_pm - f_mp + f_mm) / (4 * h[j] * h[k])
                H[j, k] = mixed_partial
                H[k, j] = mixed_partial

        H *= (-m / 2)
        covariance_matrix = np.linalg.pinv(H, rcond=1e-10)

        _, _, estimated_noise_variance, _ = self.compute_model_probability(
            model_frequencies, model_decay_rates, estimated_noise_variance_flag=True
        )
        
        parameter_variances = np.abs(np.diag(covariance_matrix))
        model_parameter_uncertainties = np.sqrt(parameter_variances * estimated_noise_variance)

        return model_parameter_uncertainties
    

    def create_metropolis_hastings_log(self, 
                                       realized_frequencies, 
                                       realized_decay_rates, 
                                       maximum_frequency_step_size, 
                                       maximum_decay_rate_step_size, 
                                       iterations,
                                       acceptances,
                                       probabilities):
        
        initial_model_parameter_uncertainties = self.compute_model_parameter_uncertainties(self.guess_frequencies, self.guess_decay_rates)
        final_model_parameter_uncertainties = self.compute_model_parameter_uncertainties(realized_frequencies, realized_decay_rates)

        f, p = self.compute_fft(self.t, self.d, self.minimum_frequency, self.maximum_frequency)
        
        plt.figure(figsize=(10, 6))

        plt.plot(f, abs(p), color="black", label="FFT")
        plt.vlines([self.guess_frequencies], -1, max(abs(p)) * 2, colors="cornflowerblue", linestyles="dashed", label="FFT Frequencies")
        plt.vlines([realized_frequencies], -1, max(abs(p)) * 2, colors="lightcoral", linestyles="dashed", label="Bayesian Frequencies")

        plt.xlim(min(f), max(f))
        plt.ylim(0, max(abs(p)) * 1.1)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power")
        plt.title("Fourier Space Model Comparison")

        plt.tight_layout()
        plt.legend()
        plt.savefig("BATS_frequency_space_model.png", dpi=300)


        # 1. Split the uncertainties into their frequency and decay rate halves
        n_freqs = len(self.guess_frequencies)
        
        initial_freq_unc = initial_model_parameter_uncertainties[:n_freqs]
        initial_decay_unc = initial_model_parameter_uncertainties[n_freqs:]
        
        final_freq_unc = final_model_parameter_uncertainties[:n_freqs]
        final_decay_unc = final_model_parameter_uncertainties[n_freqs:]

        # 2. Build the updated Frequency Table (Expanded to 89 characters wide)
        frequency_table = f"{'Index':<8} | {'Guess Frequency':<18} | {'Initial Unc.':<18} | {'Realized Frequency':<18} | {'Final Unc.':<18}\n"
        frequency_table += "-" * 89 + "\n"
        
        # zip() now pairs up guess, realized, and both their uncertainties
        for i, (guess, realized, guess_unc, real_unc) in enumerate(zip(self.guess_frequencies, realized_frequencies, initial_freq_unc, final_freq_unc)):
            frequency_table += f"{i:<8} | {guess:<18.10f} | {guess_unc:<18.10f} | {realized:<18.10f} | {real_unc:<18.10f}\n"

        # 3. Build the updated Decay Rate Table (Expanded to 90 characters wide)
        decay_rate_table = f"{'Index':<8} | {'Guess Decay Rate':<18} | {'Initial Unc.':<18} | {'Realized Decay Rate':<18} | {'Final Unc.':<18}\n"
        decay_rate_table += "-" * 90 + "\n"
        
        for i, (guess, realized, guess_unc, real_unc) in enumerate(zip(self.guess_decay_rates, realized_decay_rates, initial_decay_unc, final_decay_unc)):
            decay_rate_table += f"{i:<8} | {guess:<18.10f} | {guess_unc:<18.10f} | {realized:<18.10f} | {real_unc:<18.10f}\n"

        acceptance_rate = (acceptances / iterations) * 100 if iterations > 0 else 0

        _, _, guess_estimated_noise_variance, guess_SNR = self.compute_model_probability(self.guess_frequencies, self.guess_decay_rates, estimated_noise_variance_flag=True, SNR_flag=True, save_model_flag=True, model_title="BATS_initial_model")
        _, _, realized_estimated_noise_variance, realized_SNR = self.compute_model_probability(realized_frequencies, realized_decay_rates, estimated_noise_variance_flag=True, SNR_flag=True, save_model_flag=True, model_title="BATS_final_model")

        with open("BATS.txt", "w") as file:
            log_content = f"""Metropolis-Hastings Log
  
Frequency Information
{frequency_table}

Decay Rate Information
{decay_rate_table}

Maximum frequency step size: {maximum_frequency_step_size:<18.12f}
Maximum decay rate step size: {maximum_decay_rate_step_size:<18.12f}

Initial probability: {probabilities[0]}
Final probability: {probabilities[-1]}

Initial SNR: {guess_SNR}
Final SNR: {realized_SNR}

Initial variance: {guess_estimated_noise_variance}
Final variance: {realized_estimated_noise_variance}

Acceptance rate: {acceptance_rate}
"""
            file.write(log_content)  


    def run(self, sampling_method="metropolis_hastings", iterations=None, maximum_frequency_step_size=None, maximum_decay_rate_step_size=None, cooling=False):
        current_frequencies = self.guess_frequencies.copy()
        current_decay_rates = self.guess_decay_rates.copy()

        if sampling_method == "metropolis_hastings":
            current_probability, _, _, _ = self.compute_model_probability(current_frequencies, current_decay_rates)

            if maximum_frequency_step_size is None:
                maximum_frequency_step_size = 0.0001 * np.mean(current_frequencies)
            if maximum_decay_rate_step_size is None:
                maximum_decay_rate_step_size = 0.0001 * np.mean(current_decay_rates)
            if iterations is None:
                iterations = 10_000

            temperature = 1.00
            
            if cooling == True:
                cooling_rate = 0.01 ** (1 / iterations)
            else:
                cooling_rate = 1

            acceptances = 0
            probabilities = np.zeros(iterations)

            progress = tqdm.tqdm(total=iterations)
            for iteration in range(iterations):
                proposal_frequencies = [current_frequency + random.uniform(-maximum_frequency_step_size, maximum_frequency_step_size) for current_frequency in current_frequencies]
                proposal_decay_rates = [current_quality_factor + random.uniform(-maximum_decay_rate_step_size, maximum_decay_rate_step_size) for current_quality_factor in current_decay_rates]

                proposal_probability, _, _, _ = self.compute_model_probability(proposal_frequencies, proposal_decay_rates)

                a = (proposal_probability - current_probability) / temperature

                c = np.log10(random.random() + 1e-100)

                if c < a:
                    current_frequencies = proposal_frequencies
                    current_decay_rates = proposal_decay_rates

                    current_probability = proposal_probability

                    acceptances += 1

                probabilities[iteration] = current_probability
                temperature *= cooling_rate

                progress.update(1)

            self.create_metropolis_hastings_log(current_frequencies, current_decay_rates, maximum_frequency_step_size, maximum_decay_rate_step_size, iterations, acceptances, probabilities)
