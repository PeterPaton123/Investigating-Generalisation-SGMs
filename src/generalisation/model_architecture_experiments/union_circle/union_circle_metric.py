from jax import vmap
from jax.tree_util import Partial
import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance
import jax.scipy.stats

def split_union(samples):
    left_mask = samples[:, 0] < 0
    right_mask = ~left_mask
    left_union = samples[left_mask, :]
    right_union = samples[right_mask, :]
    return left_union, right_union

def translate(left_union, left_translate, right_union, right_translate):
    left_union_translate = vmap(lambda x: x + left_translate)(left_union)
    right_union_translate = vmap(lambda x: x + right_translate)(right_union)
    return left_union_translate, right_union_translate

def distance_simple_union(reserved_train_data, generated_samples, alpha=2):
    """
    Treats all as uniform distributions, assuming the reserved training set is representative of the manifold this should be a sufficient metric
    """
    reserved_left, reserved_right = split_union(reserved_train_data)
    reserved_left_translate, reserved_right_translate = translate(reserved_left, jnp.array([2, 0]), reserved_right, jnp.array([-2, 0]))
    generated_left, generated_right = split_union(generated_samples)
    generated_left_translate, generated_right_translate = translate(generated_left, jnp.array([2, 0]), generated_right, jnp.array([-2, 0]))
    reserved = jnp.vstack((reserved_left_translate, reserved_right_translate))
    generated = jnp.vstack((generated_left_translate, generated_right_translate))

    reserved_angles = jnp.arctan2(reserved[:, 1], reserved[:, 0])
    generated_angles = jnp.arctan2(generated[:, 1], generated[:, 0])
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return wasserstein_distance(u_values=reserved_angles, v_values=generated_angles) + alpha * wasserstein_distance(u_values=reserved_radii, v_values=generated_radii)

def distance_circle(reserved_train_data, generated_samples):
    """
    Next we model the distribution from the reserved training data and we calculate the wassenstein distance from the model distribution and our samples
    The radius is modelled as a one dimensional Gaussian with parameters around 0 and 1
    The angle is uniform between the maximum and minimum observed
    """
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    radii_mu = jnp.mean(reserved_radii)
    radii_std = jnp.std(reserved_radii)
    
    reserved_angles = jnp.arctan2(reserved_train_data[:, 1], reserved_train_data[:, 0])
    reserved_angles_min = jnp.min(reserved_angles)
    reserved_angles_max = jnp.max(reserved_angles)

    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0])

    cdf_interval = jnp.linspace(0.001, 0.999, 51)
    radii_inverse_cdf = Partial(jax.scipy.stats.norm.ppf, loc=radii_mu, scale=radii_std)
    radii_pdf = Partial(jax.scipy.stats.norm.pdf, loc=radii_mu, scale=radii_std)
    interval = vmap(radii_inverse_cdf)(cdf_interval)
    pdf_interval = vmap(radii_pdf)(interval)

    return wasserstein_distance(u_values=interval, v_values=generated_radii, u_weights=pdf_interval) + wasserstein_distance(u_values=jnp.linspace(reserved_angles_min, reserved_angles_max, 50), v_values=generated_angles)

def distance_true_union(generated_samples):
    """
    The true manifold is radius kroniker delta around 1 (for which the wassenstein distance to any point is |p-1|)
    True distribution: Radius is Kroniker delta at 1, angle is a discrete distribution of angles
    The wassenstein distance_1 between point distributions is |x-p|
    """
    generated_left, generated_right = split_union(generated_samples)
    generated_left_translate, generated_right_translate = translate(generated_left, jnp.array([2, 0]), generated_right, jnp.array([-2, 0]))
    generated = jnp.vstack((generated_left_translate, generated_right_translate))
    generated_angles = jnp.arctan2(generated[:, 1], generated[:, 0])
    generated_radii = vmap(jnp.linalg.norm)(generated)
    return wasserstein_distance(generated_angles, jnp.linspace(0, 2*jnp.pi, 100)) + jnp.mean(jnp.abs(generated_radii - 1))