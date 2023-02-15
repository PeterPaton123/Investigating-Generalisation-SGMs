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

def main():
    num_epochs = 15000
    rng = random.PRNGKey(2023)
    rng, step_rng, sample_rng = random.split(rng, 3)
    num_samples = 8

    samples_half = sample_circle(num_samples, -2, 0, sample_rng, offset=False)
    samples_half_2 = sample_circle(num_samples, 2, 0, sample_rng, offset=False)
    samples = jnp.vstack((samples_half, samples_half_2))

    samples_2_half = sample_circle(num_samples, -2, 0, sample_rng, offset=True)
    samples_2_half_2 = sample_circle(num_samples, 2, 0, sample_rng, offset=True)
    samples_2 = jnp.vstack((samples_2_half, samples_2_half_2))

    plot_samples(samples=samples, index=(0, 1), fname="samples_train", lims=((-2, 2), (-2, 2)))
    plot_samples(samples=samples_2, index=(0, 1), fname="samples_test", lims=((-2, 2), (-2, 2)))
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
    generalisation_metric_simple = np.zeros(15)
    generalisation_metric_true = np.zeros(15)
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
        plot_score(score=trained_score, t=0.01, area_min=-4, area_max=4, fname=f"bin/trained score epoch-{1000*(i+1)}")
        sampler = EulerMaruyama(sde, trained_score).get_sampler(stack_samples=False)
        q_samples = sampler(rng, n_samples=1000, shape=(N,))
        generalisation_metric_simple[i] = distance_simple_union(samples_2, q_samples)
        generalisation_metric_true[i] = distance_true_union(q_samples)
        plot_heatmap(samples=q_samples[:, [0, 1]], area_min=-4, area_max=4, fname=f"bin/heatmap trained score-{1000*(i+1)}")
    #print(generalisation_metrics)
    print(generalisation_metric_simple)
    #print(generalisation_metric)
    print(generalisation_metric_true)
    plt.plot(jnp.array(range(15)), jnp.exp(jnp.array(generalisation_metric_simple)), color='r', label="Simple")
    #plt.plot(jnp.array(range(15)), jnp.exp(jnp.array(generalisation_metric)), color='g', label="Metric")
    plt.plot(jnp.array(range(15)), jnp.exp(jnp.array(generalisation_metric_true)), color='b', label="True")
    plt.legend(loc="upper right")
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
