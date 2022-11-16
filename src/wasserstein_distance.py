import scipy
import numpy as np
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

def ws_dist_two_samples(sample1, sample2):
    """Wasserstein distance between two samples."""
    return scipy.stats.wasserstein_distance(sample1, sample2, u_weights=None, v_weights=None)

def ws_dist_sample_pdf(sample, pdf):
    return ws_dist_two_samples(sample, np.random.choice(sample, size=jnp.size(sample), p=pdf/np.sum(pdf)))

def ws_dist_normal(samples, mean, var): 
    """Wasserstein distance to standard normal"""
    n = np.shape(samples)[0]
    ## Calculates the inverse cdf of the standard normal distribution
    true_inv_cdf = jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, n)[1:-1], loc=mean, scale=np.sqrt(var))
    approx_inverse_cdf = jnp.sort(samples)[1:-1]
    return np.mean(jnp.abs(true_inv_cdf - approx_inverse_cdf))

def ws_dist_normal_mixture(samples, ws, ms, vs):
    n = np.shape(samples)[0]
    true_inv_cdf = 0
    for i in range(np.size(ws)):
        sd = np.sqrt(vs[i])
        true_inv_cdf += ws[i] * jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, n)[1:-1], loc=ms[i], scale=sd)
    approx_inverse_cdf = jnp.sort(samples)[1:-1]
    a = np.mean(jnp.abs(true_inv_cdf - approx_inverse_cdf))
    if (not np.isfinite(a)):
        print("ERROROR 2")
    return a
