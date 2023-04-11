import jax.numpy as jnp
from jax import vmap
from scipy.stats import wasserstein_distance

def swirl_metric_simple(reserved_train_data, generated_samples):
    reserved_angles = jnp.arctan2(reserved_train_data[:, 1], reserved_train_data[:, 0]) + jnp.pi
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    reserved_radii = vmap(jnp.linalg.norm)(reserved_train_data)
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return jnp.array([wasserstein_distance(reserved_angles, generated_angles), wasserstein_distance(reserved_radii, generated_radii)])