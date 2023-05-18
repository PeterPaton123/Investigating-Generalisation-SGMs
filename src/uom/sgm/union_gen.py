import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import make_union_circle, make_circle
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp
import numpy as np
from math import prod
import flax.linen as nn

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

train_data = make_circle(512, -4, 0)
train_data_2 = make_circle(512, 4, 0)

def plot(generated_samples, epoch):
    fig, axs = plt.subplots(1, 1, figsize=(10, 4.5))
    fig.suptitle("SGM generated samples")
    axs.set_ylim(-2, 2)
    axs.set_xlim(-6, 6)
    axs.set_aspect('equal')
    axs.set_facecolor('#EBEBEB')
    axs.plot(train_data[:, 0], train_data[:, 1], c='k', linestyle="--")
    axs.plot(train_data_2[:, 0], train_data_2[:, 1], c='k', linestyle="--")
    axs.scatter(generated_samples[:, 0], generated_samples[:, 1], s=3, alpha=1.0, zorder=10)
    axs.grid(which='major', color='white', linewidth=0.8)
    axs.grid(which='major', color='white', linewidth=0.8)
    fig.savefig(
        f"bin-2/{epoch}-generated",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)

class MLP_(nn.Module):
    """
    A simple model with multiple fully connected layers and some fourier features
    for the time variable. Functions are designed for a mini-batch of inputs.
    """
    layers = (16, 32, 64, 128, 256)
    @nn.compact
    def __call__(self, x, t):
        x_shape = x.shape
        in_size = prod(x_shape[1:])
        t = t.reshape((t.shape[0], -1))
        x = x.reshape((x.shape[0], -1))
        t = jnp.concatenate([t - 0.5, jnp.cos(2*jnp.pi*t)], axis=-1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers:
            x = nn.Dense(layer)(x)
            x = nn.relu(x)
        x = nn.Dense(in_size)(x)
        return x.reshape(x_shape)
    
    def evaluate(self, params, x_t, times):
        return self.apply(params, x_t, times)

samples_half = make_circle(16, -4, 0)
samples_half_2 = make_circle(16, 4, 0)
train_samples =  jnp.vstack((samples_half, samples_half_2))
rng = random.PRNGKey(2023)
NUM = 30
num_epochs = NUM * 1000
rng, step_rng = random.split(rng, 2)
N = train_samples.shape[1]
# Get sde model
sde = OU(beta_max=40.0)
# Neural network training via score matching
score_model = MLP_()
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
generalisation_metric_true = np.zeros(NUM, dtype=object)
for i in range(NUM):
# Train with score matching
    score_model, params, opt_state, mean_losses = retrain_nn(
        update_step=update_step,
        num_epochs=1_000,
        step_rng=step_rng,
        samples=train_samples,
        score_model=score_model,
        params=params,
        opt_state=opt_state,
        loss=loss,
        batch_size=batch_size)
    # Get trained score
    trained_score = get_score(sde, score_model, params, score_scaling=True)
    sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
    q_samples = sampler(rng, num_samples=2_000, shape=(N,))
    plot(q_samples, (i+1)*1_000)