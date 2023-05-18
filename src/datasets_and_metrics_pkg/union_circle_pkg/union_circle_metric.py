import jax.numpy as jnp
from jax import vmap
from scipy.stats import wasserstein_distance

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

def true_circle_metric(generated_samples):
    samples_x_less_0 = generated_samples[:, 0] < 0
    generated_angles = jnp.arctan2(generated_samples[:, 1], generated_samples[:, 0]) + jnp.pi
    generated_radii = vmap(jnp.linalg.norm)(generated_samples)
    return (wasserstein_distance(generated_angles, jnp.linspace(0, 2*jnp.pi, 100)), jnp.mean(jnp.abs(generated_radii - 1))) # jnp.sqrt(jnp.mean((generated_radii - 1)**2))

def union_circle_metric(generated_samples):
    """
    Returns 3 quantities:
    How well the angles are distributed of the generated samples
    How well the radii are distributed
    How well distributed (ideally evenly) are the samples between the two manifolds
    """
    generated_left, generated_right = split_union(generated_samples)
    generated_left_translate, generated_right_translate = translate(generated_left, jnp.array([2, 0]), generated_right, jnp.array([-2, 0]))
    generated = jnp.vstack((generated_left_translate, generated_right_translate))
    generated_angles = jnp.arctan2(generated[:, 1], generated[:, 0]) + jnp.pi
    generated_radii = vmap(jnp.linalg.norm)(generated)
    return jnp.array([wasserstein_distance(generated_angles, jnp.linspace(0, 2*jnp.pi, 100)), jnp.mean(jnp.abs(generated_radii - 1)), wasserstein_distance(u_values=(generated_samples[:, 0] < 0), v_values=[0, 1])])
