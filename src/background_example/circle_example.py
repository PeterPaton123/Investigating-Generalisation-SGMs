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
    xs = x0 + jnp.cos(alphas)
    ys = y0 + jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    #samples += 0.05 * random.normal(key, shape=jnp.shape(samples))
    return samples

def main():
    CONST = 30
    num_epochs = CONST * 1000
    rng = random.PRNGKey(2023)
    rng, step_rng, sample_rng = random.split(rng, 3)
    num_samples = 8
    samples = sample_circle(num_samples, 0, 0, sample_rng, offset=False)
    """
    index = (0, 1)
    fname="samples"
    lims=((-2, 2), (-2, 2))
    fig, ax = plt.subplots(1, 1)
    fig.patch.set_facecolor('white')
    fig.patch.set_alpha(1.0)
    circle = patches.Circle((0, 0), radius=1, fc=(0, 0, 0, 0), ec='blue', linestyle='--', label="Target")
    ax.add_patch(circle)
    ax.scatter(
        samples[:, index[0]], samples[:, index[1]],
        color='red', label="Train")
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

    sde = OU()
    """
    def log_hat_pt(x, t):
        mean, std = sde.marginal_prob(samples, t)
        potentials = jnp.sum(-(x - mean)**2 / (2 * std**2), axis=1)
        return logsumexp(potentials, axis=0, b=1/num_samples)
    rng, *sample_rng = random.split(rng, 2)
    # Get a jax grad function, which can be batched with vmap
    nabla_log_hat_pt = jit(vmap(grad(log_hat_pt), in_axes=(0, 0), out_axes=(0)))
    plot_score(score=nabla_log_hat_pt, t=0.01, area_min=-2, area_max=2, fname="bin/generalisation-example-empirical-score")
    sampler = EulerMaruyama(sde, nabla_log_hat_pt).get_sampler(stack_samples=False)
    q_samples = sampler(rng, n_samples=3000, shape=(N,))    
    plot_heatmap(samples=q_samples, area_min=-2, area_max=2, fname="bin/generalisation-example-heatmap-empirical-score")
    # Get sde model
    """
    """
    sde = OU()
    CONST = 30

    # Neural network training via score matching
    batch_size=16
    score_model = MLP()
    # Initialize parameters
    params = score_model.init(step_rng, jnp.zeros((batch_size, N)), jnp.ones((batch_size,)))
    # Initialize optimizer
    opt_state = optimizer.init(params)
    # Get loss function
    loss = get_loss_fn(
        sde, score_model, score_scaling=True, likelihood_weighting=False,
        reduce_mean=True, pointwise_t=False)
    generalisation_metric_simple = np.empty(CONST, dtype=object)
    generalisation_metric = np.empty(CONST, dtype=object)
    generalisation_metric_true = np.empty(CONST, dtype=object)
    for i in range(CONST):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/CONST),
            step_rng=step_rng,
            samples=samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss_fn=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
        sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
        q_samples = sampler(rng, n_samples=3000, shape=(N,))    
        plot_heatmap(samples=q_samples, area_min=-2, area_max=2, fname=f"bin/generalisation-example-heatmap-trained-score-epoch-{1000*(i+1)}")
        plot_score(score=trained_score, t=0.01, area_min=-2, area_max=2, fname=f"bin/generalisation-example-trained-score-epoch-{1000*(i+1)}")
    """
    
if __name__ == "__main__":
    main()
