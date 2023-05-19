import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import sliced_wasserstein
from sklearn.datasets import make_swiss_roll
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

def run_experiment(train_samples, test_samples, beta, rng):
    N = 30
    num_epochs = N * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=beta, beta_min=beta)
    # Neural network training via score matching
    score_model = MLP()
    batch_size = 128
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
        local_plot(q_samples, f"results/{beta}/samples-{1000*(i+1)}-epochs.png")
        generalisation_metric_train[i] = sliced_wasserstein(train_samples, q_samples, 100)
        generalisation_metric_true[i] = sliced_wasserstein(test_samples, q_samples, 100)
    pd.DataFrame(generalisation_metric_train).to_csv(f"results/{beta}/true.csv", index=None, header=None)
    pd.DataFrame(generalisation_metric_true).to_csv(f"results/{beta}/true.csv", index=None, header=None)

def local_plot(generated_samples, fname):
    fig = plt.figure(figsize=plt.figaspect(1.0/2))
    ax_0 = fig.add_subplot(1, 1, 1, projection='3d')
    ax_0.scatter(generated_samples[:, 0], generated_samples[:, 1], generated_samples[:, 2], c='k', s=10, alpha=0.8)
    ax_0.set_title("Generated Swiss Roll")
    ax_0.view_init(azim=-66, elev=12)
    fig.savefig(fname)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    betas = np.logspace(-2, 2, 10)
    sr_samples, sr_color = make_swiss_roll(n_samples=7000, random_state=0)
    train_indices = np.array([i for i in range(7000) if i % 5 == 0])
    true_indices = np.array([i for i in range(7000) if i % 5 != 0])
    for beta in betas:
        run_experiment(sr_samples[train_indices], sr_samples[true_indices], beta, rng)
