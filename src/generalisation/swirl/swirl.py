"""Score based generative models introduction.

Based off the Jupyter notebook: https://jakiw.com/sgm_intro
A tutorial on the theoretical and implementation aspects of score-based generative models, also called diffusion models.
"""
from jax import jit, vmap, grad
import numpy as np
from jax.lax import dynamic_slice
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_animation)
from diffusionjax.losses import get_loss_fn
from diffusionjax.samplers import EulerMaruyama
from diffusionjax.utils import (
    MLP,
    CNN,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.sde import OU

"""
Initialise our samples
"""
def sample_swirl(num_samples, rngkey, offset=False):
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + offset * (jnp.pi/num_samples * jnp.ones(num_samples))
    rs = 0.5 + jnp.cos(alphas)
    xs, ys = rs * jnp.cos(alphas) + jnp.ones(num_samples), rs * jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def main():
    num_epochs = 15000
    rng = random.PRNGKey(2023)
    rng, step_rng, sample_rng = random.split(rng, 3)
    num_samples = 14
    samples = sample_swirl(num_samples, sample_rng, offset=False)
    samples_2 = sample_swirl(num_samples, sample_rng, offset=True)
    plot_samples(samples=samples, index=(0, 1), fname="samples_train", lims=((-2, 2), (-2, 2)))
    plot_samples(samples=samples_2, index=(0, 1), fname="samples_test", lims=((-2, 2), (-2, 2)))
    """
    index = (0, 1)
    fname="samples_test"
    lims=((-0.4, 2), (-1.2, 1.2))
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    ax.scatter(
        samples[:, index[0]], samples[:, index[1]],
        color='red', label="Train")
    samples = samples_2
    ax.scatter(
        samples[:, index[0]], samples[:, index[1]],
        color='green', label="Test")
    samples = sample_swirl(400, sample_rng, False)
    ax.plot(
        samples[:, index[0]], samples[:, index[1]],
        color='k', linestyle='--', label="Target")
    ax.legend()
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
    N = samples.shape[1]

    # Get sde model
    sde = OU()

    # Neural network training via score matching
    batch_size=16
    score_model = MLP() # MLP_simple(num_neurons_per_layer=jnp.array([256, 256, 256]))
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Get loss function
    loss = get_loss_fn(
        sde, score_model, score_scaling=True, likelihood_weighting=False,
        reduce_mean=True, pointwise_t=False)
    for i in range(15):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/15),
            step_rng=step_rng,
            samples=samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss_fn=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
        plot_score(score=trained_score, t=0.01, area_min=-1.2, area_max=3, fname=f"bin/trained score epoch-{1000*(i+1)}")
        sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
        q_samples = sampler(rng, n_samples=1000, shape=(N,))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-1.2, area_max=3, fname=f"bin/heatmap trained score-{1000*(i+1)}")
    frames = 100
    fig, ax = plt.subplots()
    def animate(i, ax):
        ax.clear()
        plot_score_ax(
            ax, trained_score, t=1 - (i / frames), area_min=-3, area_max=3, fname="trained score")
    # Plot animation of the trained score over time
    plot_animation(fig, ax, animate, frames, "bin/trained_score_anim")

if __name__ == "__main__":
    main()
