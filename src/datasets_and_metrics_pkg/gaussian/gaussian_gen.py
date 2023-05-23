import numpy as np
import jax.numpy as jnp
from jax import vmap
import jax.random as random
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import multivariate_normal, norm, wasserstein_distance

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
    
    def pdf(self, x):
        res = np.zeros(x.shape[0])
        for i in range(self.n_components):
            res += multivariate_normal.pdf(x, mean=self.mus[i], cov=self.cholensky_decompositions[i]@(self.cholensky_decompositions[i]).T) * (1.0 / self.n_components)
        return res
    
    def one_dim_xs(self):
        N = 50
        xs = np.empty(N * self.n_components)
        for i in range(self.n_components):
            us = np.linspace(0.005, 0.995, N)
            xs[i*N:(i+1)*N] = norm.ppf(us, loc=self.mus[i], scale=self.cholensky_decompositions[i])
        return xs
    
    """
    One dimensional distance
    """
    def one_dimensional_wasserstein(self, generated_samples):
        return wasserstein_distance(u_values=self.one_dim_xs(), v_values=generated_samples, u_weights=self.pdf(self.one_dim_xs()))
    
if __name__ == '__main__':
    mus = jnp.array([[-5.], [5.]])
    covars = jnp.array([np.eye(1), np.eye(1)])
    gmm = GMM(mus, covars)
    plt.hist(gmm.sample(300), bins=50, color = np.repeat('b', 300), stacked=True, density=True)
    plt.plot(gmm.one_dim_xs(), gmm.pdf(gmm.one_dim_xs()), c='r', linestyle='--')
    plt.show()