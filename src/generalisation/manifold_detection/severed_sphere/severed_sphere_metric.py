import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.stats import wasserstein_distance
from severed_sphere_samples import ss_sample_gen
from plot import plot
import matplotlib.pyplot as plt
"""
Training samples are constructed in the following way:
xs = jnp.sin(t_samples) * jnp.cos(p_samples)
ys = jnp.sin(t_samples) * jnp.sin(p_samples)
zs = jnp.cos(t_samples)
"""

def ss_metric(generated_samples, p_reduce, t_reduce=jnp.pi/9):
    """
    Generalisation metric for the severed sphere compares the sampled values after transformation as trigonometric functions are not 1 to 1
    """
    p_true = np.linspace(start=0, stop=2*jnp.pi - p_reduce, num=10_000)
    t_true = np.linspace(start = t_reduce, stop = (jnp.pi - t_reduce), num=10_000)

    #wasserstein_distance(generated_samples[:, 1] / generated_samples[:, 0], jnp.tan(p_true))
    return wasserstein_distance(jnp.sin(t_true) * jnp.cos(p_true), generated_samples[:, 0]), wasserstein_distance(jnp.sin(t_true) * jnp.sin(p_true), generated_samples[:, 1]), wasserstein_distance(jnp.cos(t_true), generated_samples[:, 2])

if __name__ == "__main__":
    rng = random.PRNGKey(4)
    rng, sample_rng = random.split(rng, 2)
    samples, colour = ss_sample_gen(num_samples=2_500, p_reduce=0.55, sample_rng=sample_rng)
    plot(samples, fname="test", num_neighbours=25, colours=colour)
    #print(ss_metric(samples, p_reduce=0.55))