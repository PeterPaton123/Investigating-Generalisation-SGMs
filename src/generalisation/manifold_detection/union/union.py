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

import sys
sys.path.append('/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/generalisation/model_architecture_experiments/')
from models import (
    MLP3L16N,
    MLP3L64N,
    MLP3L256N,
    MLP5L16N,
    MLP5L64N,
    MLP5L256N,
)

def sample_circle_filled(num_samples, sample_rng, x0=0, y0=0):
    """
    Sample in 2d
    """
    radius_rng, angle_rng = random.split(sample_rng, 2)
    radii = jnp.sqrt(random.uniform(radius_rng, shape=(num_samples,), dtype=float, minval=0, maxval=0.5))
    alphas = random.uniform(angle_rng, shape=(num_samples,), dtype=float, minval=0, maxval=2 * jnp.pi * (1 - 1/num_samples))
    xs = radii * jnp.cos(alphas) + x0
    ys = radii * jnp.sin(alphas) + y0
    samples = jnp.stack([xs, ys], axis=1)
    # Project into 3d
    return samples

def union_run(score_model, train_samples, rng):
    NUM = 15
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
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
    for i in range(NUM):
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/NUM),
            step_rng=step_rng,
            samples=train_samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score(sde, score_model, params, score_scaling=True)
        plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/trained_score_epoch_{1000*(i+1)}")
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=2_000, shape=(N,))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-4, area_max=4, fname=f"bin/heatmap_trained_score-{1000*(i+1)}")

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    rng, sample_rng = random.split(rng, 2)
    # Sample generation
    num_samples_1 = 500
    samples_half = np.zeros((num_samples_1, 2))
    samples_half[:, 0] = np.linspace(-np.pi, 0, num_samples_1)
    samples_half[:, 1] = 0.5 * np.sin(3 * samples_half[:, 0])
    samples_half_2 = sample_circle_filled(1500, sample_rng, x0=2, y0=0)
    train_samples = jnp.vstack((samples_half, samples_half_2))
    
    union_run(MLP3L16N(), train_samples, rng)
    """
    index = (0, 1)
    fname="true_samples"
    lims=((-3.5, 3.5), (-2, 2))
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.scatter(
        train_samples[:, index[0]], train_samples[:, index[1]],
        color='Blue', s=5)
    ax.set_xlabel(r"$x_{}$".format(index[0]))
    ax.set_ylabel(r"$x_{}$".format(index[1]))
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    fig.savefig(
        fname,
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    """
