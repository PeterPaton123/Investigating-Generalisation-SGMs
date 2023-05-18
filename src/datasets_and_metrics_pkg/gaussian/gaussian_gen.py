import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax.random as random
import matplotlib.pyplot as plt
import pandas as pd

class GMM:
    """
    Gaussian mixture model
    Currently this is equally weights between components (an easy change if we require otherwise however)
    """
    def __init__(self, mus, covars):
        """
        Input:
        mus: Numpy array of the means (N_components, dim)
        covars: numpy array of the covariance matrices (N_components, dim, dim)
        """
        (self.n_components, self.dim) = jnp.shape(mus)
        self.mus = jnp.array(mus)
        self.cholensky_decompositions = vmap(jnp.linalg.cholesky)(jnp.array(covars))
        
    def sample(self, n_samples, rng=random.PRNGKey(0)):
        mixture_samples = jnp.empty((0, self.dim), dtype=jnp.float32)
        component_rng, sample_rng = random.split(rng, 2)
        sampled_components = random.choice(component_rng, jnp.arange(start=0, stop=self.n_components, step=1), shape=(n_samples, ))
        samples = random.multivariate_normal(sample_rng, mean=jnp.zeros(self.dim), cov=jnp.eye(self.dim), shape=(n_samples, ))
        components, counts = jnp.unique(sampled_components, size=self.n_components, fill_value=0, return_counts=True)
        on_going_count = 0
        for component, count in zip(components, counts):
            component_samples = self.mus[component] + jnp.matmul(self.cholensky_decompositions[component], samples[on_going_count:count+on_going_count].T).T
            mixture_samples = jnp.vstack((mixture_samples, component_samples))
            on_going_count += count
        return mixture_samples
    