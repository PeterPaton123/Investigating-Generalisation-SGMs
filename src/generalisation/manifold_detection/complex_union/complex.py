from sklearn.datasets import make_swiss_roll
from sklearn.manifold import SpectralEmbedding
from sklearn import manifold
import matplotlib.pyplot as plt
from jax import jit, vmap, grad
import numpy as np
from jax.lax import dynamic_slice
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_animation)
from diffusionjax.losses import get_loss
from diffusionjax.solvers import EulerMaruyama
from diffusionjax.samplers import (
    get_sampler
)
from diffusionjax.utils import (
    get_score,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.models import (
    MLP,
    CNN)
from diffusionjax.sde import OU

rng = random.PRNGKey(2023)
rng, sample_rng = random.split(rng, 2)

def sample_circle_filled_3d(num_samples, sample_rng, x0=0, y0=0):
    """
    Sample in 3d
    """
    radius_rng, angle_rng = random.split(sample_rng, 2)
    radii = jnp.sqrt(random.uniform(radius_rng, shape=(num_samples,), dtype=float, minval=0, maxval=4.0))
    alphas = random.uniform(angle_rng, shape=(num_samples,), dtype=float, minval=0, maxval=2 * jnp.pi * (1 - 1/num_samples))
    xs = radii * jnp.cos(alphas) + x0
    ys = radii * jnp.sin(alphas) + y0
    zs = -10 * np.ones(len(alphas))
    samples = jnp.stack([xs, ys, zs], axis=1)
    # Project into 3d
    return samples

sr_samples, sr_color = make_swiss_roll(n_samples=2000, random_state=0, hole=True)
train_samples_2 = sample_circle_filled_3d(1500, sample_rng, 10, 5)
samples = jnp.vstack([sr_samples, train_samples_2])
c = np.concatenate([sr_color, 10 * np.ones(1500)])

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    samples[:, 0], samples[:, 1], samples[:, 2], c=c, s=10, alpha=0.8
)
ax.view_init(azim=-66, elev=12)
fig.savefig(f"complex_union.png")

NUM = 15
num_epochs = NUM * 1000
rng, step_rng = random.split(rng, 2)
N = sr_samples.shape[1]
score_model = MLP()
# Get sde model
sde = OU(beta_max=20.0)
# Neural network training via score matching
batch_size=128
# Initialize parameters
params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
# Initialize optimizer
opt_state = optimizer.init(params)
# Initalize solver
solver = EulerMaruyama(sde)
# Get loss function
loss = get_loss(
    sde, solver, score_model, score_scaling=True, likelihood_weighting=False,
    reduce_mean=True, pointwise_t=False)
manifold_metric = np.zeros(NUM, dtype=object)
mmd_metric = np.zeros(NUM, dtype=object)
for i in range(NUM):
# Train with score matching
    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step=update_step,
        num_epochs=int(num_epochs/NUM),
        step_rng=step_rng,
        samples=samples,
        score_model=score_model,
        params=params,
        opt_state=opt_state,
        loss=loss,
        batch_size=batch_size)
    # Get trained score
    trained_score = get_score(sde, score_model, params, score_scaling=True)
    #plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/trained_score_epoch_{1000*(i+1)}")
    sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
    q_samples = sampler(rng, num_samples=3500, shape=(N,))
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    fig.add_axes(ax)
    ax.scatter(
        q_samples[:, 0], q_samples[:, 1], q_samples[:, 2], c='k', s=10, alpha=0.8
    )
    ax.set_title("Generated Swiss Roll")
    ax.view_init(azim=-66, elev=12)
    fig.savefig(f"bin/heatmap_samples_{(i+1)*1000}.png")

    