
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import jax.random as random
import jax.numpy as jnp
from vae import VAE

def sample_circle_filled(num_samples, sample_rng, x0=0, y0=0):
    """
    Sample in 2d
    """
    radius_rng, angle_rng = random.split(sample_rng, 2)
    radii = jnp.sqrt(random.uniform(radius_rng, shape=(num_samples,), dtype=float, minval=0, maxval=0.5))
    alphas = random.uniform(angle_rng, shape=(num_samples,), dtype=float, minval=0, maxval=2 * jnp.pi * (1 - 1/num_samples))
    xs = radii * jnp.cos(alphas) + x0
    ys = radii * jnp.sin(alphas) + y0
    samples = np.stack([xs, ys], axis=1)
    # Project into 3d
    return samples

rng = random.PRNGKey(2023)
rng, sample_rng = random.split(rng, 2)
# Sample generation
num_samples_1 = 10_000
samples_half = np.zeros((num_samples_1, 2))
samples_half[:, 0] = np.linspace(-np.pi, 0, num_samples_1)
samples_half[:, 1] = 0.5 * np.sin(3 * samples_half[:, 0])
samples_half_2 = sample_circle_filled(10_000, sample_rng, x0=2, y0=0)
samples_union = np.vstack((samples_half, samples_half_2))

sinusoidal_vae = VAE()
filled_circle_vae = VAE()
union_vae = VAE()

sinusoidal_vae.train(samples_half)
filled_circle_vae.train(samples_half_2)
union_vae.train(samples_union)

sinusoidal_vae.generate_samples(num_samples=200, fname="bin/sinusoidal_generated_samples.png")
filled_circle_vae.generate_samples(num_samples=200, fname="bin/filled_circle_generated_samples.png")
union_vae.generate_samples(num_samples=200, fname="bin/union_generated_samples.png")
