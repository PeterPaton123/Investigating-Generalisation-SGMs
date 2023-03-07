"""Score based generative models introduction.

Based off the Jupyter notebook: https://jakiw.com/sgm_intro
A tutorial on the theoretical and implementation aspects of score-based generative models, also called diffusion models.
"""
from random import sample
from jax import jit, vmap, random
import jax.numpy as jnp
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from numpy import size
plt.style.use('seaborn-poster')

"""
Produces samples inside the unit circle in 2d
"""
def sample_circle_filled(num_samples, sample_rng, x0=0, y0=0):
    """
    Sample in 2d
    """
    radius_rng, angle_rng = random.split(sample_rng, 2)
    radii = jnp.sqrt(random.uniform(radius_rng, shape=(num_samples,), dtype=float, minval=0, maxval=1.0))
    alphas = random.uniform(angle_rng, shape=(num_samples,), dtype=float, minval=0, maxval=2 * jnp.pi * (1 - 1/num_samples))
    xs = radii * jnp.cos(alphas) + x0
    ys = radii * jnp.sin(alphas) + y0
    samples = jnp.stack([xs, ys], axis=1)
    # Project into 3d
    return samples

def observed_samples(num_samples, sample_rng, A=jnp.eye(3)):
    latent_samples = sample_circle_filled(num_samples, sample_rng)
    return (A@latent_samples.T).T

def main():
    rng = random.PRNGKey(42)
    rng, train_sample_rng, test_sample_rng = random.split(rng, 3)
    fig_2d, axs_2d = plt.subplots(1)
    latent_train_samples_2d = sample_circle_filled(500, train_sample_rng)
    latent_test_samples_2d = sample_circle_filled(500, test_sample_rng)
    axs_2d.scatter(latent_train_samples_2d[:, 0], latent_train_samples_2d[:, 1], marker='.', alpha=0.8, color="tab:blue", label="Training set")
    axs_2d.scatter(latent_test_samples_2d[:, 0], latent_test_samples_2d[:, 1], marker='.', alpha=0.8, color="tab:red", label="Test set")
    axs_2d.legend(loc="upper right")
    axs_2d.set_title("Latent samples in 2d")
    fig_2d.show()

    fig_3d = plt.figure(figsize = (10,10))
    axs_3d = plt.axes(projection='3d')
    eps = 1e-16
    axs_3d.axes.set_xlim3d(left=-1.5-eps, right=1.5+eps)
    axs_3d.axes.set_ylim3d(bottom=-1.5-eps, top=1.5+eps) 
    axs_3d.axes.set_zlim3d(bottom=-2.-eps, top=2+eps)

    projection_observed = jnp.array([[1, 0], [0, 1], [1, 1]])
    observed_train_samples = (projection_observed@latent_train_samples_2d.T).T
    observed_test_samples = (projection_observed@latent_test_samples_2d.T).T

    axs_3d.scatter(observed_train_samples[:, 0], observed_train_samples[:,1], observed_train_samples[:, 2], color="tab:blue", label="Training set")
    axs_3d.scatter(observed_test_samples[:, 0], observed_test_samples[:,1], observed_test_samples[:, 2], color="tab:red", label="Test set")
    axs_3d.set_title("Observed samples in 3D space")
    axs_3d.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    main()