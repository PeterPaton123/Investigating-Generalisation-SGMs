import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import make_severed_sphere, plot_severed_sphere
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


def run_experiment(prop_samples, rng):
    rng, sample_rng, sample_rng_2 = random.split(rng, 3)
    num_samples = int(prop_samples * 2_500)
    train_samples, train_colour = make_severed_sphere(num_samples, sample_rng, p_reduce=0.15)
    test_samples, test_colour = make_severed_sphere(5_000, sample_rng_2, p_reduce=0.15)
    plot_severed_sphere(train_samples, fname=f"bin/{prop_samples}/_setup", colours=train_colour)
    NUM = 15
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=20.0)
    # Neural network training via score matching
    score_model = MLP()
    batch_size=num_samples
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
    generalisation_metric_sliced_ws = np.zeros(NUM, dtype=object)
    generalisation_metric_ss = np.zeros(NUM, dtype=object)
    for i in range(NUM):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=1000,
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
        plot_severed_sphere(samples=q_samples, fname=f"bin/{prop_samples}/generated_samples_{int((num_epochs/10))*(i+1)}")
        #generalisation_metric_simple[i] = distance_simple_circle(test_samples, q_samples)
        #generalisation_metric_true[i] = distance_true_circle(q_samples)
    #pd.DataFrame(generalisation_metric_simple).to_csv(f"results/{num_samples}/simple.csv", index=None, header=None)
    #pd.DataFrame(generalisation_metric_true).to_csv(f"results/{num_samples}/true.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    rng, sample_rng = random.split(rng, 2)
    for prop_samples in [1, 0.5, 0.3, 0.2, 0.1]:
        run_experiment(prop_samples, rng)


