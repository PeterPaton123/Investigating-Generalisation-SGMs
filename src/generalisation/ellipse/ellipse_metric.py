from jax import vmap
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance
import jax.scipy.stats

import matplotlib.pyplot as plt

def inverse_transformation(data, A):
    """
    The data is assumed to be data = Ax, this function returns x
    """
    return (jnp.linalg.inv(A.T@A)@A.T@(data.T)).T

def distance_simple_ellipse(reserved_train_data, generated_samples, alpha=2.0, A=jnp.eye(2)):
    """
    Treats all as uniform distributions, assuming the reserved training set is representative of the manifold this should be a sufficient metric
    """
    reserved_train_data = inverse_transformation(reserved_train_data, A)
    generated_samples = inverse_transformation(generated_samples, A)
    reserved_angles = jnp.arctan2(reserved_train_data[:, 1], reserved_train_data[:, 0]) + jnp.pi
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return (wasserstein_distance(u_values=reserved_angles, v_values=generated_angles), alpha * wasserstein_distance(u_values=reserved_radii, v_values=generated_radii))

def distance_ellipse(reserved_train_data, generated_samples, A=jnp.eye(2)):
    """
    Next we model the distribution from the reserved training data and we calculate the wassenstein distance from the model distribution and our samples
    The radius is modelled as a one dimensional Gaussian with parameters around 0 and 1
    The angle is uniform between the maximum and minimum observed
    """
    reserved_train_data = inverse_transformation(reserved_train_data, A)
    generated_samples = inverse_transformation(generated_samples, A)
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

def distance_true_ellipse(generated_samples, A=jnp.eye(2)):
    """
    The true manifold is radius kroniker delta around 1 (for which the wassenstein distance to any point is |p-1|)
    True distribution: Radius is Kroniker delta at 1, angle is a discrete distribution of angles
    The wassenstein distance_1 between point distributions is |x-p|
    """
    generated_samples = inverse_transformation(generated_samples, A)
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return (wasserstein_distance(generated_angles, jnp.linspace(0, 2*jnp.pi, 100)), jnp.mean(jnp.abs(generated_radii - 1)))