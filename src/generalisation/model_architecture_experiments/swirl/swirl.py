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
from swirl_metric import swirl_metric_simple
from models import (
    MLP3L16N,
    MLP3L64N,
    MLP3L256N,
    MLP5L16N,
    MLP5L64N,
    MLP5L256N,
    MLP5L512N,
)

"""
Initialise our samples
"""
def sample_swirl(num_samples, rngkey, offset=False):
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + offset * (jnp.pi/num_samples * jnp.ones(num_samples))
    rs = 0.5 + jnp.cos(alphas)
    xs, ys = rs * jnp.cos(alphas), rs * jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def inner_visualise(samples, samples_2, true_samples):
    index = (0, 1)
    fname="true_samples"
    lims=((-0.4, 2), (-1.2, 1.2))
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.plot(
        true_samples[:, index[0]], true_samples[:, index[1]],
        color='b', linestyle='--', label="Target")
    ax.scatter(
        samples[:, index[0]], samples[:, index[1]],
        color='red', label="Train")
    ax.scatter(
        samples_2[:, index[0]], samples_2[:, index[1]],
        color='green', label="Test")
    ax.set_xlabel(r"$x_{}$".format(index[0]))
    ax.set_ylabel(r"$x_{}$".format(index[1]))
    ax.legend()
    if lims is not None:
        ax.set_xlim(lims[0])
        ax.set_ylim(lims[1])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    fig.savefig(
        fname,
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def swirl_run(score_model, train_samples, test_samples, rng, name):
    NUM = 30
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = samples.shape[1]
    # Get sde model
    sde = OU(beta_max=3.0)
    # Neural network training via score matching
    batch_size=16
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Initalize solver
    solver = EulerMaruyama(sde)
    # Get loss function
    loss = get_loss(
        sde, solver, score_model, score_scaling=True, likelihood_weighting=False, reduce_mean=True, pointwise_t=False)
    metrics = np.empty(NUM, dtype=object)
    for i in range(NUM):
    # Train with score matching
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
        plot_score(score=trained_score, t=0.01, area_min=-1.2, area_max=1.8, fname=f"bin/trained_score_epoch-{1000*(i+1)}")
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=10_000, shape=(N,))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-1.2, area_max=1.8, fname=f"bin/heatmap_trained_score-{1000*(i+1)}")
        metrics[i] = swirl_metric_simple(test_samples, q_samples)
    pd.DataFrame(metrics).to_csv(f"results/{name}.csv", index=None, header=None)

if __name__ == "__main__":
    num_samples = 40
    rng = random.PRNGKey(2023)
    rng, sample_rng = random.split(rng, 2)
    samples = sample_swirl(num_samples, sample_rng, offset=False)
    true_samples = sample_swirl(400, sample_rng, offset=True)
    models = [MLP5L512N()] #[MLP3L16N(), MLP3L64N(), MLP3L256N(), MLP5L16N(), MLP5L64N(), MLP5L256N()]
    model_names = ["5L512N"]#["3L16N", "3L64N", "3L256N", "5L16N", "5L64N", "5L256N"]
    for model, name in zip(models, model_names):
        swirl_run(model, samples, true_samples, rng, name)