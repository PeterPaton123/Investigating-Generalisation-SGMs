from jax import random
import jax.numpy as jnp

def make_severed_sphere(num_samples, sample_rng, p_reduce = 0.55, t_reduce=jnp.pi/8):
    """
    Severed sphere sample generation
    """
    t_rng, p_rng = random.split(sample_rng, 2)
    t_samples = random.uniform(t_rng, shape=(num_samples,), dtype=float, minval=t_reduce, maxval=jnp.pi - t_reduce)
    p_samples = random.uniform(p_rng, shape=(num_samples,), dtype=float, minval=0, maxval=2 * jnp.pi - p_reduce)
    colours = p_samples
    # Sever the poles from the sphere.
    xs = jnp.sin(t_samples) * jnp.cos(p_samples)
    ys = jnp.sin(t_samples) * jnp.sin(p_samples)
    zs = jnp.cos(t_samples)
    samples = jnp.stack([xs, ys, zs], axis=1)
    return samples, colours