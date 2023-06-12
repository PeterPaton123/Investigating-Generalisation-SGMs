import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import GMM
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
    CNN,
    get_score_fn,
    update_step,
    optimizer,
    retrain_nn)
from diffusionjax.sde import OU

def dummy_score(x, t):
    return 0.0


def beta_experiment(beta, mu):
    rng = random.PRNGKey(42)
    rng, step_rng, sample_rng = random.split(rng, 3)
    # Sample generation
    mus = jnp.array([[-mu], [mu]])
    covars = jnp.array([np.eye(1), np.eye(1)])
    gmm = GMM(mus, covars)
    samples = gmm.sample(2_000, rng=sample_rng)

    sde = OU(beta_min=beta, beta_max=beta, n_steps=1000)
    sampler = EulerMaruyama(sde, dummy_score).get_sampler(stack_samples=False)
    q_samples = sampler(rng, n_samples=3000, shape=(N,))



for beta in betas:
    run_experiment(gmm, beta, rng)


