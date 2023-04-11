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

from union_circle_metric import distance_simple_union, distance_true_union

"""
Initialise our samples
"""
def sample_circle(num_samples, x0, y0, rngkey, offset=False):
    """Samples from the unit circle centered at x0 y0, angles split equally around point.
    Offset will rotate the samples and generate different samples along the same manifold

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + offset * (jnp.pi/num_samples * jnp.ones(num_samples))
    xs = jnp.ones(num_samples) * x0 + jnp.cos(alphas)
    ys = jnp.ones(num_samples) * y0 + jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    #samples += 0.05 * random.normal(key, shape=jnp.shape(samples))
    return samples

def union_run(score_model, train_samples, test_samples, rng, name):
    NUM = 30
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=20.0)
    # Neural network training via score matching
    batch_size=8
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
    generalisation_metric_simple = np.zeros(NUM, dtype=object)
    generalisation_metric_true = np.zeros(NUM, dtype=object)
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
        #plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/trained_score_epoch_{1000*(i+1)}")
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=10_000, shape=(N,))
        generalisation_metric_simple[i] = distance_simple_union(test_samples, q_samples)
        generalisation_metric_true[i] = distance_true_union(q_samples)
    pd.DataFrame(generalisation_metric_simple).to_csv(f"results/simple-{name}.csv", index=None, header=None)
    pd.DataFrame(generalisation_metric_true).to_csv(f"results/true-{name}.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    rng, sample_rng = random.split(rng, 2)
    rng = random.PRNGKey(2023)
    rng, sample_rng = random.split(rng, 2)
    # Sample generation
    num_samples = 8
    samples_half = sample_circle(num_samples, -2, 0, sample_rng, offset=False)
    samples_half_2 = sample_circle(num_samples, 2, 0, sample_rng, offset=False)
    train_samples = jnp.vstack((samples_half, samples_half_2))
    samples_2_half = sample_circle(num_samples, -2, 0, sample_rng, offset=True)
    samples_2_half_2 = sample_circle(num_samples, 2, 0, sample_rng, offset=True)
    test_samples = jnp.vstack((samples_2_half, samples_2_half_2))
    
    true_samples_half = sample_circle(100, -2, 0, sample_rng, offset=True)
    true_samples_half_2 = sample_circle(100, 2, 0, sample_rng, offset=True)
    true_samples = jnp.vstack((true_samples_half, true_samples_half_2))
    
    models = [MLP3L16N(), MLP3L64N(), MLP3L256N(), MLP5L16N(), MLP5L64N(), MLP5L256N()]
    model_names = ["3L16N", "3L64N", "3L256N", "5L16N", "5L64N", "5L256N"]
    for model, name in zip(models, model_names):
        union_run(model, train_samples, test_samples, rng, name)
    