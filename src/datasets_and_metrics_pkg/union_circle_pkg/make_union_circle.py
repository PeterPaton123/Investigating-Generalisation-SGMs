import jax.numpy as jnp

def make_circle(num_samples, x0, y0, offset=False):
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + offset * (jnp.pi/num_samples * jnp.ones(num_samples))
    xs = x0 + jnp.cos(alphas)
    ys = y0 + jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def make_union_circle(num_samples, offset=False):
    samples_half = make_circle(num_samples, -2, 0, offset=offset)
    samples_half_2 = make_circle(num_samples, 2, 0, offset=offset)
    return jnp.vstack((samples_half, samples_half_2))