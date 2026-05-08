import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks
import jax
import jax.numpy as jnp
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
        
        self.t = t
        self.d = d

        self.minimum_frequency = minimum_frequency

        if maximum_frequency is None:
            self.maximum_frequency = jnp.mean(jnp.diff(t))
        else:
            self.maximum_frequency = maximum_frequency

        if guess_frequencies is None:
            f, p = self.get_fft(t, d, minimum_frequency=minimum_frequency, maximum_frequency=maximum_frequency)
            self.guess_frequencies, _ = self.get_peaks(f, abs(p))
            self.guess_frequencies = self.guess_frequencies.tolist()
        else:
            self.guess_frequencies = guess_frequencies.tolist()

        if guess_decay_rates is None:
            self.guess_decay_rates = self.get_decay_rates(t, d, self.guess_frequencies)

        if signals is None:
            self.signals = len(self.guess_frequencies)
        else:
            self.signals = signals
            self.guess_frequencies = self.guess_frequencies[:signals]
            self.guess_decay_rates = self.guess_decay_rates[:signals]

        self.decay_type = decay_type


    def get_fft(self, t, d, minimum_frequency=None, maximum_frequency=None):
        delta = np.mean(np.diff(t)) 

        taper = signal.get_window(('kaiser', 2. * np.pi), len(d))
        d = d * taper
        
        nfft = 2 ** (math.ceil(math.log(len(d), 2)) + 2)
        frequencies = np.fft.fftfreq(n=nfft, d=delta)
        power = np.fft.fft(d, n=nfft) * delta

        mask = np.ones_like(frequencies, dtype=bool)

        if minimum_frequency is not None:
            mask &= (frequencies >= minimum_frequency) 

        if maximum_frequency is not None:
            mask &= (frequencies <= maximum_frequency)

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

            peak_t, peak_d = self.get_peaks(t, frequency_d)

            res = stats.linregress(np.array(peak_t), np.array(np.log(peak_d)))

            decay_rates[ind] = -1 * res.slope

        return decay_rates.tolist()


    def get_model_probability(self, model_frequencies, model_decay_rates, estimated_noise_variance_flag=False, SNR_flag=False):
        omegas = np.array(model_frequencies) * 2 * np.pi
        alphas = np.array(model_decay_rates)
        
        t = self.t
        d = self.d

        r = len(omegas)
        m = 2 * r
        N = len(d)
        
        arg = omegas[:, None] * t[None, :]
        decay = np.exp(-alphas[:, None] * t[None, :])
        
        G_cos = np.cos(arg) * decay
        G_sin = np.sin(arg) * decay
        
        G = np.vstack((G_cos, G_sin))
        g = G @ G.T

        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / np.sqrt(eigenvalues)[None, :]
        H = scaled_eigenvectors.T @ G

        h = H @ d

        mean_square_data = (1 / N) * np.sum(d ** 2)
        mean_square_projection = (1 / m) * np.sum(h ** 2)

        ratio = (m * mean_square_projection) / (N * mean_square_data)
        log_probability = 0.5 * (m - N) * np.log(max(1 - ratio, 1e-15))

        estimated_noise_variance = None
        if estimated_noise_variance_flag == True or SNR_flag == True:
            estimated_noise_variance = (1 / (N - m - 2)) * (np.sum(d ** 2) - np.sum(h ** 2))

        SNR = None
        if SNR_flag == True:
            SNR = ((m / N) * (1 + mean_square_projection / estimated_noise_variance)) ** (0.5)

        return log_probability, mean_square_projection, estimated_noise_variance, SNR
    

    def get_model_probability_jax(self, model_frequencies, model_decay_rates, t, d):
        # 1. Convert inputs to JAX arrays
        omegas = jnp.array(model_frequencies) * 2 * jnp.pi
        alphas = jnp.array(model_decay_rates)
        
        r = len(omegas)
        m = 2 * r
        N = len(d)
        
        arg = omegas[:, None] * t[None, :]
        decay = jnp.exp(-alphas[:, None] * t[None, :])
        
        G_cos = jnp.cos(arg) * decay
        G_sin = jnp.sin(arg) * decay
        
        G = jnp.vstack((G_cos, G_sin))
        g = G @ G.T

        # 2. JAX handles eigendecompositions perfectly
        eigenvalues, eigenvectors = jnp.linalg.eigh(g)
        eigenvalues = jnp.maximum(eigenvalues, 1e-8)

        scaled_eigenvectors = eigenvectors / jnp.sqrt(eigenvalues)[None, :]
        H = scaled_eigenvectors.T @ G

        h = H @ d

        mean_square_data = (1 / N) * jnp.sum(d ** 2)
        mean_square_projection = (1 / m) * jnp.sum(h ** 2)

        ratio = (m * mean_square_projection) / (N * mean_square_data)
        
        # 3. CRITICAL FIX: Use jnp.maximum instead of built-in max()
        log_probability = 0.5 * (m - N) * jnp.log(jnp.maximum(1 - ratio, 1e-15))

        # JAX grad requires the function to return the scalar we are differentiating FIRST.
        # We return the projection as "auxiliary" data.
        return log_probability, mean_square_projection


    def get_model_probability_gradient(self, q):
        r = len(q) // 2
        frequencies = np.array(q[:r])
        alphas = np.array(q[r:])
        omegas = frequencies * 2 * np.pi
        
        t = self.t
        d = self.d
        N = len(d)
        m = 2 * r
        
        # 1. Forward Pass Components (recalculated for the current q state)
        arg = omegas[:, None] * t[None, :]
        decay = np.exp(-alphas[:, None] * t[None, :])
        
        G_cos = np.cos(arg) * decay
        G_sin = np.sin(arg) * decay
        G = np.vstack((G_cos, G_sin))
        
        g = G @ G.T
        eigenvalues, eigenvectors = np.linalg.eigh(g)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        
        scaled_eigenvectors = eigenvectors / np.sqrt(eigenvalues)[None, :]
        H = scaled_eigenvectors.T @ G
        h = H @ d
        
        # 2. Compute Intermediate Vectors for the Gradient
        # c represents the coefficients of the basis functions: c = (G G^T)^-1 G d
        # Using the eigendecomposition, this simplifies elegantly to:
        c = scaled_eigenvectors @ h 
        
        # P_d is the projection of the data onto the basis: P_d = G^T c
        P_d = H.T @ h
        
        # The residual of the fit
        residual = d - P_d
        
        # 3. Vectorized Basis Derivatives
        # Pre-weighting the basis rows by 't' to satisfy the derivative chain rule
        t_G_cos = t * G_cos
        t_G_sin = t * G_sin
        
        c_cos = c[:r]
        c_sin = c[r:]
        
        # Gradients of the mean square projection (S) w.r.t parameters.
        # Matrix multiplication handles the summation across time points automatically.
        dS_df = (4 * np.pi / m) * (c_sin[:, None] * t_G_cos - c_cos[:, None] * t_G_sin) @ residual
        dS_dalpha = (-2 / m) * (c_cos[:, None] * t_G_cos + c_sin[:, None] * t_G_sin) @ residual
        
        # 4. Apply Chain Rule to the Final Log Probability
        sum_sq_data = np.sum(d ** 2)
        sum_sq_proj = np.sum(h ** 2)
        
        # The chain factor maps dS/dq to dLogProb/dq analytically
        denominator = max(sum_sq_data - sum_sq_proj, 1e-15)
        chain_factor = (m * (N - m)) / (2 * denominator)
        
        dL_df = chain_factor * dS_df
        dL_dalpha = chain_factor * dS_dalpha
        
        # Return as a single concatenated flat array matching the shape of 'q'
        return np.concatenate((dL_df, dL_dalpha))
    

    def run(self, B, M, L, epsilon=None, maximum_frequency_p=None, maximum_decay_rate_p=None):
        if maximum_frequency_p is None:
            maximum_frequency_p = 0.01 * np.array(self.guess_frequencies)

        if maximum_decay_rate_p is None:
            maximum_decay_rate_p = 0.01 * np.array(self.guess_decay_rates)

        q_0 = np.concatenate([self.guess_frequencies, self.guess_decay_rates])
        parameters = len(q_0)
        n_half = parameters // 2

        if epsilon is None:
            epsilon = 1e-6

        initial_scales = np.abs(q_0) * 0.10
        initial_scales = np.maximum(initial_scales, 1e-6)
        
        # We will use M_mass to represent the diagonal mass matrix (1/variance)
        M_mass = 1.0 / (initial_scales ** 2)

        # ---------------------------------------------------------
        # 1. JAX SETUP: Create a JIT-compiled objective function
        # ---------------------------------------------------------
        # Convert static data to JAX arrays once to avoid overhead
        t_jax = jnp.array(self.t)
        d_jax = jnp.array(self.d)

        # Define a wrapper that takes a single vector `q` and returns JUST the log prob scalar
        def objective_fn(q):
            freqs = q[:n_half]
            decays = q[n_half:]
            log_prob, _ = self.get_model_probability_jax(freqs, decays, t_jax, d_jax)
            return log_prob

        # jax.value_and_grad computes both the log_prob AND its gradient in one pass.
        # jax.jit compiles it to C++/CUDA for massive speedups.
        val_and_grad_fn = jax.jit(jax.value_and_grad(objective_fn))

        # ---------------------------------------------------------

        q_history = []
        probabilities = []
        B_acceptances = 0

        # Initial evaluation
        current_log_probability, gradient = val_and_grad_fn(q_0)
        current_log_probability = np.array(current_log_probability)
        gradient = np.array(gradient)

        # ==========================================
        # BURN-IN PHASE
        # ==========================================
        for step in tqdm(range(B), desc="Burn-in Samples"):
            # Draw momentum using the current mass
            p = np.random.normal(0, np.sqrt(M_mass), parameters)
            T_0 = 0.5 * np.sum(p ** 2 / M_mass)

            q = q_0.copy()
            p_current = p.copy()
            grad_current = gradient.copy()

            # --- Vectorized Leapfrog Integrator ---
            for _ in range(L):
                # Half step for momentum
                p_current = p_current + (epsilon / 2.0) * grad_current
                
                # Full step for position
                q = q + epsilon * (p_current / M_mass)
                
                # Vectorized Reflective Boundary
                out_of_bounds = q < 1e-10
                q = np.where(out_of_bounds, 2e-10 - q, q)  # Equivalent to 1e-10 + (1e-10 - q)
                p_current = np.where(out_of_bounds, -p_current, p_current)

                # Get new gradient
                _, grad_current = val_and_grad_fn(q)
                grad_current = np.array(grad_current)

                # Half step for momentum
                p_current = p_current + (epsilon / 2.0) * grad_current
            # --------------------------------------

            # Evaluate proposal
            proposal_log_probability, _ = val_and_grad_fn(q)
            proposal_log_probability = np.array(proposal_log_probability)
            
            T = 0.5 * np.sum(p_current ** 2 / M_mass)
            
            # Calculate Hamiltonian (H = -log_prob + Kinetic Energy)
            H_0 = -current_log_probability + T_0
            H = -proposal_log_probability + T

            # Metropolis Acceptance
            a = H_0 - H
            c = np.log(random.random() + 1e-100)

            if c < a:
                q_0 = q
                current_log_probability = proposal_log_probability
                gradient = grad_current # Save gradient for next step
                B_acceptances += 1

            q_history.append(q_0.copy())
            probabilities.append(current_log_probability)

            # # --- Mass Matrix Adaptation ---
            if step > B // 2:
                samples = np.array(q_history[int(B/2):])
                m_scales = np.std(samples, axis=0)
                min_scales = np.abs(q_0) * 0.01
                m_scales = np.maximum(m_scales, min_scales)
                M_mass = 1.0 / (m_scales**2)

        print(f"Burn-in Acceptance Rate: {B_acceptances / B:.4f}")
        
        # ==========================================
        # SAMPLING PHASE
        # ==========================================
        acceptances = 0
        q_history = []
        
        for _ in tqdm(range(M), desc="Sampling"):
            p = np.random.normal(0, np.sqrt(M_mass), parameters)
            T_0 = 0.5 * np.sum(p ** 2 / M_mass)

            q = q_0.copy()
            p_current = p.copy()
            grad_current = gradient.copy()

            # --- Vectorized Leapfrog Integrator ---
            for _ in range(L):
                p_current = p_current + (epsilon / 2.0) * grad_current
                
                q = q + epsilon * (p_current / M_mass)
                
                out_of_bounds = q < 1e-10
                q = np.where(out_of_bounds, 2e-10 - q, q)
                p_current = np.where(out_of_bounds, -p_current, p_current)

                _, grad_current = val_and_grad_fn(q)
                grad_current = np.array(grad_current)

                p_current = p_current + (epsilon / 2.0) * grad_current
            # --------------------------------------

            proposal_log_probability, _ = val_and_grad_fn(q)
            proposal_log_probability = np.array(proposal_log_probability)
            
            T = 0.5 * np.sum(p_current ** 2 / M_mass)

            H_0 = -current_log_probability + T_0
            H = -proposal_log_probability + T

            a = H_0 - H
            c = np.log(random.random() + 1e-100)

            if c < a:
                q_0 = q
                current_log_probability = proposal_log_probability
                gradient = grad_current
                acceptances += 1

            q_history.append(q_0.copy())
            probabilities.append(current_log_probability)
        
        # ==========================================
        # REPORTING & VISUALIZATION
        # ==========================================
        initial_state = q_history[0]
        final_state = q_history[-1]

        print(f"Sampling Acceptance Rate: {acceptances / M:.4f}")

        print(f"{'Index':<7} | {'Initial Value':<15} | {'Final Value':<15} | {'Difference':<15}")
        print("-" * 62)

        for i, (init, final) in enumerate(zip(initial_state, final_state)):
            diff = final - init
            print(f"{i:<7} | {init:<15.10f} | {final:<15.10f} | {diff:<+15.10f}")

        # Re-evaluate with your custom flags for the final printout
        # Assuming you still want to call the original get_model_probability for these specific printouts
        probability_0 = self.get_model_probability(q_history[0][:n_half], q_history[0][n_half:], estimated_noise_variance_flag=True, SNR_flag=True)
        probability = self.get_model_probability(q_history[-1][:n_half], q_history[-1][n_half:], estimated_noise_variance_flag=True, SNR_flag=True)
        probability_new = self.get_model_probability(q_history[-1][:n_half], q_history[-1][n_half:], estimated_noise_variance_flag=True, SNR_flag=True)

        old_probability = self.get_model_probability([0.0003044944,0.0003066753,0.0003089359,0.0003112798,0.0003136873], 
                                                     [0.0000019319,0.0000019313,0.0000019311,0.0000019312,0.0000019314], 
                                                     estimated_noise_variance_flag=True, SNR_flag=True)

        print("Prob 0:", probability_0)
        print("Prob Final:", probability)
        print("Prob Old:", old_probability)

        plt.plot(probabilities)
        plt.title("Log Probability over HMC Steps")
        plt.xlabel("Step")
        plt.ylabel("Log Probability")
        plt.show()
