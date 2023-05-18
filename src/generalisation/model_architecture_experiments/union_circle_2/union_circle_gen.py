import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import make_union_circle, union_circle_metric
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

from models import (
    MLP3L16N,
    MLP3L64N,
    MLP3L256N,
    MLP5L16N,
    MLP5L64N,
    MLP3L12N,
)

def plot_local(train, test, name):
    index = (0, 1)
    lims=((-3.5, 3.5), (-2, 2))
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    true_samples = make_union_circle(500)
    true_samples_left_mask = true_samples[:, 0] < 0
    ax.plot(
        true_samples[true_samples_left_mask, index[0]], true_samples[true_samples_left_mask, index[1]],
        color='b', linestyle='--', label="Target")
    ax.plot(
        true_samples[~true_samples_left_mask, index[0]], true_samples[~true_samples_left_mask, index[1]],
        color='b', linestyle='--')
    ax.scatter(
        train[:, index[0]], train[:, index[1]],
        color='red', label="Train")
    ax.scatter(
        test[:, index[0]], test[:, index[1]],
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
        f"bin/{name}/_setup.png",
        facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()

def run_experiment(score_model, rng, name):
    num_samples=8
    train_samples = make_union_circle(num_samples, offset=False)
    test_samples = make_union_circle(num_samples, offset=True)
    plot_local(train_samples, test_samples, name)
    NUM = 30
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=20.0)
    # Neural network training via score matching
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
        plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/{name}/trained_score_epoch_{(int(num_epochs/NUM))*(i+1)}.png")
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=2500, shape=(N,))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-4, area_max=4, fname=f"bin/{name}/heatmap_trained_score_{int((num_epochs/NUM))*(i+1)}.png")
        generalisation_metric_true[i] = union_circle_metric(q_samples)
    pd.DataFrame(generalisation_metric_true).to_csv(f"results/{name}/true.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    models = [MLP3L12N(), MLP3L16N(), MLP3L64N(), MLP3L256N(), MLP5L16N(), MLP5L64N()]
    model_names = ["3L12N", "3L16N", "3L64N", "3L256N", "5L16N", "5L64N"]
    for score_model, name in zip(models, model_names):
        run_experiment(score_model, rng, name)