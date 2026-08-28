import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsp
from jax.scipy.special import logsumexp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

@jax.jit
def log_prob(t, d, fs, ks):
    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Calculate the ratio of the squared projections to the squared data.
    # Note: The m and N terms cancel out here compared to calculating the 
    # explicit mean square data (msd) and mean square projection (msp).
    sum_sq_data = jnp.sum(d ** 2)
    sum_sq_proj = jnp.sum(h ** 2)

    ratio = sum_sq_proj / sum_sq_data
    
    return 0.5 * (m - N) * jnp.log(1.0 - ratio)


@jax.jit
def get_model(t, d, fs, ks):
    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    model = jnp.zeros(len(H))
    for ind, _ in enumerate(h):
        model += h[ind] * H[ind]

    return model


def bats_model(t, d, f_loc, f_scale, k_loc, k_scale):
    # Gaussian priors centered at f_loc / k_loc
    fs = numpyro.sample("fs", dist.Normal(f_loc, f_scale).to_event(1))
    
    # If ks represents decay or rate that must stay strictly non-negative,
    # consider dist.TruncatedNormal(k_loc, k_scale, low=0.0)We
    ks = numpyro.sample("ks", dist.Normal(k_loc, k_scale).to_event(1))
    
    # Custom likelihood factor
    numpyro.factor("bretthorst", log_prob(t, d, fs, ks))


def statistics(t, d, fs, ks):
    omegas = fs * 2.0 * jnp.pi
    
    r = omegas.shape[0]
    m = 2 * r
    N = d.shape[0]

    arg = omegas[:, None] * t[None, :]
    decay = jnp.exp(-ks[:, None] * t[None, :])

    # Build the non-orthogonal model matrix G and its Gram matrix
    G = jnp.vstack((jnp.cos(arg) * decay, jnp.sin(arg) * decay))
    gram = G @ G.T

    # Eigendecomposition for orthogonalization
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 1e-12)

    # Bretthorst Eq. 3.6: orthonormal functions H
    H = (eigenvectors / jnp.sqrt(eigenvalues)).T @ G
    
    # Bretthorst Eq. 3.13: projection amplitudes h
    h = H @ d

    # Calculate the ratio of the squared projections to the squared data.
    # Note: The m and N terms cancel out here compared to calculating the 
    # explicit mean square data (msd) and mean square projection (msp).
    mean_sq_data = (1 / N) * jnp.sum(d ** 2)
    mean_sq_proj = (1 / m) * jnp.sum(h ** 2)

    ratio = (m / N) * mean_sq_proj / mean_sq_data

    prob = 0.5 * (m - N) * jnp.log(1.0 - ratio)

    # Variance --------------------------------------------------
    
    variance = (1 / (N - m - 2)) * (jnp.sum(d ** 2) - jnp.sum(h ** 2))

    # SNR -------------------------------------------------------
    
    SNR = ((m / N) * (1 + mean_sq_proj / variance)) ** (0.5)

    # Uncertainties ---------------------------------------------

    # TODO: add uncertainty calculation

    # Power Spectrum ---------------------------------------------

    f_space = jnp.append(jnp.linspace(0.98 * min(fs), 1.02 * max(fs), 10_000), fs)

    def compute_C_single(f_val):
        phase = 2 * jnp.pi * f_val * t
        return (1 / N) * jnp.abs(jnp.sum(d * jnp.exp(1j * phase))) ** 2

    C = jax.lax.map(compute_C_single, f_space)
    
    def ms_projection_wrapper(q):
        f = q[:r]
        a = q[r:]  # decay rates; assumed positive

        omega = 2.0 * jnp.pi * f

        arg = omega[:, None] * t[None, :]
        decay = jnp.exp(-a[:, None] * t[None, :])

        G = jnp.vstack((
            jnp.cos(arg) * decay,
            jnp.sin(arg) * decay
        ))

        # G G^T
        M = G @ G.T

        # Data projection
        proj_d = G @ d

        # Scale-dependent ridge for numerical stability
        ridge = 1e-8 * jnp.trace(M) / M.shape[0]
        M_reg = M + ridge * jnp.eye(M.shape[0])

        # Solve M_reg x = G d
        x = jnp.linalg.solve(M_reg, proj_d)

        # Projection power
        return jnp.dot(proj_d, x) / m

    # Combined parameter vector
    q = jnp.concatenate((
        jnp.asarray(fs),
        jnp.asarray(ks)
    ))

    hessian = jax.jit(
        jax.hessian(ms_projection_wrapper)
    )(q)

    hessian_diag = jnp.diag(hessian)[:r]
    b_diagonal = (-m / 2.0) * hessian_diag

    # p_space = (2 * (variance + jnp.sum(C)) * 
    #           jnp.sum((b_diagonal[:, None] / (2 * jnp.pi * variance)) ** (1 / 2) * 
    #           jnp.exp((-b_diagonal[:, None] * (fs[:, None] - f_space) ** 2) / (2 * variance)), axis=0))

    # 1. SAFEGUARDS: Prevent log(negative) and log(0)
    # Force b_diagonal to be strictly positive and > 0
    safe_b_diag = jnp.maximum(jnp.abs(b_diagonal), 1e-30)

    # Force variance to be > 0 to prevent division by zero or log(0)
    safe_var = jnp.maximum(variance, 1e-30)

    # 2. Compute the log of the amplitude factor using safe variables
    log_amplitude = 0.5 * (jnp.log(safe_b_diag[:, None]) - jnp.log(2 * jnp.pi * safe_var))

    # 3. The exponent uses the safe variables
    exponent = (-safe_b_diag[:, None] * (fs[:, None] - f_space) ** 2) / (2 * safe_var)

    # 4. Combine them in log space
    X = log_amplitude + exponent

    # 5. Use the LSE trick directly
    log_inner_sum = logsumexp(X, axis=0)

    # 6. Compute the log of the leading constant scalar
    # Also safeguard the sum of C just in case it dipped negative
    safe_C_sum = jnp.maximum(jnp.sum(jnp.nan_to_num(C)), 0.0)
    log_constant = jnp.log(2 * (safe_var + safe_C_sum))

    # 7. Add them together to get the final log-spectrum
    log_p_space = log_constant + log_inner_sum

    # 8. Convert back to linear space
    p_space = jnp.exp(log_p_space)
    p_spec = jnp.column_stack((f_space, p_space))

    # Global Likelihood -----------------------------------------

    R_delta = float(jnp.max(jnp.abs(d)))
    R_sigma = float(jnp.max(jnp.abs(d)))

    log_R_delta = jnp.maximum(jnp.log(R_delta), 1e-12)
    log_R_sigma = jnp.maximum(jnp.log(R_sigma), 1e-12)

    R_gamma = (0.5 / float(jnp.mean(jnp.diff(t)))) * float(t[-1] - t[0])

    b = (-m / 2.0) * hessian
    eigenvalues, _ = jnp.linalg.eigh(b)
    eigenvalues = jnp.maximum(eigenvalues, 1e-8)

    factor = ((m / 2.0) * jnp.log(2.0 * jnp.pi)
                        - 0.5 * jnp.sum(jnp.log(eigenvalues))
                        - m * jnp.log(R_gamma))
    delta_term = (jsp.gammaln(m / 2.0)
                    - jnp.log(2.0 * log_R_delta)
                    - (m / 2.0) * jnp.log((m * mean_sq_proj) / 2.0))
    sigma_term = (jsp.gammaln((N - m - r) / 2.0)
                    - jnp.log(2.0 * log_R_sigma)
                    - ((N - m - r) / 2.0) * jnp.log((N * mean_sq_data - m * mean_sq_proj) / 2.0))
    gamma_term = -(2.0 * r) * jnp.log(R_gamma)

    glob_LL = delta_term + sigma_term + gamma_term + factor

    return [prob, variance, SNR, p_spec, glob_LL]


class BATS():
    def __init__(self, 
                 t, 
                 d, 
                 f_init, 
                 k_init
                 ):
        
        self.t = jnp.array(t)
        self.d = jnp.array(d)
        
        self.f_init = jnp.array(f_init)
        self.k_init = jnp.array(k_init)


    def run_NUTS(self, 
                 f_bw, 
                 k_bw, 
                 W, 
                 S,
                 seed
                 ):

        f_bw = jnp.array(f_bw)
        k_bw = jnp.array(k_bw)
        
        init_strategy = init_to_value(values={"fs": self.f_init, "ks": self.k_init})

        kernel = NUTS(
            bats_model,
            init_strategy=init_strategy,
            dense_mass=True,
            target_accept_prob=0.8,
            max_tree_depth=8
        )

        mcmc = MCMC(
            kernel,
            num_warmup=W,
            num_samples=S,
            num_chains=1,
            progress_bar=True
        )

        mcmc.run(
            jax.random.PRNGKey(seed),
            self.t,
            self.d,
            self.f_init,
            f_bw,
            self.k_init,
            k_bw,
            extra_fields=("potential_energy",)
        )

        samples = mcmc.get_samples()

        pe = mcmc.get_extra_fields()["potential_energy"]
        best_ind = jnp.argmin(pe)

        best_fs = samples["fs"][best_ind]
        best_ks = samples["ks"][best_ind]

        return best_fs, best_ks
