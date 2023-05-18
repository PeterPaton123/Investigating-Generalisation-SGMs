import jax.numpy as jnp

def make_circle(num_samples, x0, y0, offset=False):
    """
    Samples from the unit circle centered at x0 y0, angles split equally around point.
    Offset will rotate the samples and generate different samples along the same manifold

    Args:
        num_samples: The number of samples.
        x0, y0 coordinates of the center
        offset: a further rotation applied to the samples to separate test and train data
    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + offset * (jnp.pi/num_samples * jnp.ones(num_samples))
    xs = x0 + jnp.cos(alphas)
    ys = y0 + jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples