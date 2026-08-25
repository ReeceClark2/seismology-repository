import jax
import jax.numpy as jnp

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


def bats_model(t, d, f_loc, f_scale, k_loc, k_scale):
    # Gaussian priors centered at f_loc / k_loc
    fs = numpyro.sample("fs", dist.Normal(f_loc, f_scale))
    
    # If ks represents decay or rate that must stay strictly non-negative,
    # consider dist.TruncatedNormal(k_loc, k_scale, low=0.0)We
    ks = numpyro.sample("ks", dist.Normal(k_loc, k_scale))
    
    # Custom likelihood factor
    numpyro.factor("bretthorst", log_prob(t, d, fs, ks))


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
                 seed,
                 statistics
                 ):
        
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
