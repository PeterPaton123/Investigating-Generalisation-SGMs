import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import make_circle, distance_simple_circle, distance_true_circle
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp
import numpy as np
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

def run_experiment(train_samples, beta, rng):
    N = 30
    num_epochs = N * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=beta, beta_min=beta)
    # Neural network training via score matching
    score_model = MLP()
    batch_size = 16
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
    generalisation_metric_train = np.zeros(N, dtype=object)
    generalisation_metric_true = np.zeros(N, dtype=object)
    for i in range(N):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/N),
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
        q_samples = sampler(rng, num_samples=2500, shape=(N,))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-2, area_max=2, fname=f"results/{beta}/heatmap_trained_score_{int(1000*(i+1))}.png")
        plot_score(score=trained_score, t=0.01, area_min=-2, area_max=2, fname=f"results/{beta}/trained_score_epoch_{int(1000*(i+1))}.png")
        generalisation_metric_train[i] = distance_simple_circle(train_samples, q_samples)
        generalisation_metric_true[i] = distance_true_circle(q_samples)
    pd.DataFrame(generalisation_metric_train).to_csv(f"results/{beta}/train.csv", index=None, header=None)
    pd.DataFrame(generalisation_metric_true).to_csv(f"results/{beta}/true.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    betas = np.logspace(-2, 2, 10)
    samples = make_circle(16, 0, 0)
    for beta in betas:
        run_experiment(samples, beta, rng)
