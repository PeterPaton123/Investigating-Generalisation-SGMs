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
from diffusionjax.plot import (
    plot_samples, plot_score, plot_score_ax, plot_heatmap, plot_animation)
from diffusionjax.losses import get_loss_fn
from diffusionjax.samplers import EulerMaruyama
from diffusionjax.utils import (
    MLP,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.sde import OU

from generalisation.nnd import GeneralisationMetric

"""
Initialise our samples
"""
def sample_circle(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def sample_circle_2(num_samples):
    """Samples from the unit circle, angles split.

    Args:
        num_samples: The number of samples.

    Returns:
        An (num_samples, 2) array of samples.

    N_samples: Number of samples
    Returns a (N_samples, 2) array of samples
    """
    alphas = jnp.linspace(0, 2 * jnp.pi * (1 - 1/num_samples), num_samples) + jnp.pi/180.0 * jnp.ones(num_samples)
    xs = jnp.cos(alphas)
    ys = jnp.sin(alphas)
    samples = jnp.stack([xs, ys], axis=1)
    return samples

def main():
    num_epochs = 10000
    rng = random.PRNGKey(2023)
    rng, step_rng = random.split(rng, 2)
    num_samples = 8
    samples = sample_circle(num_samples)
    plot_samples(samples=samples, index=(0, 1), fname="samples", lims=((-3, 3), (-3, 3)))
    N = samples.shape[1]

    # Get sde model
    sde = OU()

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
    generalisation_metrics = np.zeros(10)
    for i in range(10):
    # Train with score matching
        score_model, params, opt_state, mean_losses = retrain_nn(
            update_step=update_step,
            num_epochs=int(num_epochs/10),
            step_rng=step_rng,
            samples=samples,
            score_model=score_model,
            params=params,
            opt_state=opt_state,
            loss_fn=loss,
            batch_size=batch_size)
        # Get trained score
        trained_score = get_score_fn(sde, score_model, params, score_scaling=True)
        plot_score(score=trained_score, t=0.01, area_min=-3, area_max=3, fname=f"bin/trained score epoch-{i}")
        sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
        q_samples = sampler(rng, n_samples=1000, shape=(N,))
        generalisation = GeneralisationMetric(sample_circle_2(40), q_samples)
        generalisation.train(1000)
        generalisation_metrics[i], _ = generalisation.value_and_grad_fn(generalisation.params)
        #generated_samples = jnp.append(generated_samples, q_samples, axis=0)
        #print(jnp.shape(generated_samples))
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-3, area_max=3, fname=f"bin/heatmap trained score-{i}")
    #print(generalisation_metrics)
    plt.plot(jnp.array(range(10)), jnp.array(generalisation_metrics))
    plt.show()
    
    #generalistion = GeneralisationMetric(sample_circle_2(40), generated_samples)
    #generalistion.train(1000)
    #def fun(i):
    #    mse, _ = generalistion.make_mse_loss(sample_circle_2(40), dynamic_slice(generated_samples, (i * 1000, 1), ((i+1)*1000, 1)))
    #    return mse(generalistion.params)
    
    #plt.plot(jnp.array(list(range(10))), vmap(fun)(jnp.array(list(range(10)))))
    #plt.show()

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
