from jax import vmap
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance
import jax.scipy.stats

import matplotlib.pyplot as plt


def distance_simple_circle(reserved_train_data, generated_samples, alpha=2):
    """
    Treats all as uniform distributions, assuming the reserved training set is representative of the manifold this should be a sufficient metric
    """
    reserved_angles = jnp.arctan2(reserved_train_data[:, 1], reserved_train_data[:, 0]) + jnp.pi
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return (wasserstein_distance(u_values=reserved_angles, v_values=generated_angles), alpha * wasserstein_distance(u_values=reserved_radii, v_values=generated_radii))

def distance_circle(reserved_train_data, generated_samples):
    """
    Next we model the distribution from the reserved training data and we calculate the wassenstein distance from the model distribution and our samples
    The radius is modelled as a one dimensional Gaussian with parameters around 0 and 1
    The angle is uniform between the maximum and minimum observed
    """
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    radii_mu = jnp.mean(reserved_radii)
    radii_std = jnp.std(reserved_radii)
    
    reserved_angles = jnp.arctan2(reserved_train_data[:, 1], reserved_train_data[:, 0]) + jnp.pi
    reserved_angles_min = jnp.min(reserved_angles)
    reserved_angles_max = jnp.max(reserved_angles)

    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi

    cdf_interval = jnp.linspace(0.001, 0.999, 51)
    radii_inverse_cdf = Partial(jax.scipy.stats.norm.ppf, loc=radii_mu, scale=radii_std)
    radii_pdf = Partial(jax.scipy.stats.norm.pdf, loc=radii_mu, scale=radii_std)
    interval = vmap(radii_inverse_cdf)(cdf_interval)
    pdf_interval = vmap(radii_pdf)(interval)
    return (wasserstein_distance(u_values=jnp.linspace(reserved_angles_min, reserved_angles_max, 50), v_values=generated_angles), wasserstein_distance(u_values=interval, v_values=generated_radii, u_weights=pdf_interval))

def distance_true_circle(generated_samples):
    """
    The true manifold is radius kroniker delta around 1 (for which the wassenstein distance to any point is |p-1|)
    True distribution: Radius is Kroniker delta at 1, angle is a discrete distribution of angles
    The wassenstein distance_1 between point distributions is |x-p|
    """
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return (wasserstein_distance(generated_angles, jnp.linspace(0, 2*jnp.pi, 100)), jnp.mean(jnp.abs(generated_radii - 1))) # jnp.sqrt(jnp.mean((generated_radii - 1)**2))

def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

#print(distance_circle(sample_circle(8), sample_circle(8)))
