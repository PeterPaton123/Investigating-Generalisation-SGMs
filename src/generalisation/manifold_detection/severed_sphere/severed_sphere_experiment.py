from jax import jit, vmap, grad
import numpy as np
from jax.lax import dynamic_slice
import jax.random as random
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

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

from severed_sphere_samples import ss_sample_gen
from plot import plot
from severed_sphere_metric import ss_metric

def run(rng, p_reduce):
    rng, sample_rng = random.split(rng, 2)
    # Generate samples
    train_samples, colour = ss_sample_gen(num_samples=2_500, p_reduce=p_reduce, sample_rng=sample_rng)
    plot(train_samples, fname=f"{p_reduce}/train-samples", num_neighbours=25, colours=colour)

    score_model = MLP()
    NUM = 15
    num_epochs = NUM * 1000
    rng, step_rng = random.split(rng, 2)
    N = train_samples.shape[1]
    # Get sde model
    sde = OU(beta_max=20.0)
    # Neural network training via score matching
    batch_size=128
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
    
    generalisation_metric_severed_sphere = np.zeros(NUM, dtype=object)
    generalisation_metric_full_sphere = np.zeros(NUM, dtype=object)

    for i in range(NUM):
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
        
        trained_score = get_score(sde, score_model, params, score_scaling=True)
        sampler = get_sampler(EulerMaruyama(sde.reverse(trained_score)), stack_samples=False)
        q_samples = sampler(rng, num_samples=5_000, shape=(N,))
        generalisation_metric_severed_sphere[i] = ss_metric(q_samples, p_reduce)
        generalisation_metric_full_sphere[i] = ss_metric(q_samples, 0.0)
        if i % 3 == 0:
            plot(q_samples, fname=f"{p_reduce}/generated-{(i+1)*1000}", num_neighbours=25)
    pd.DataFrame(generalisation_metric_severed_sphere).to_csv(f"results/severed-{p_reduce}.csv", index=None, header=None)
    pd.DataFrame(generalisation_metric_full_sphere).to_csv(f"results/full-{p_reduce}.csv", index=None, header=None)

if __name__ == "__main__":
    rng = random.PRNGKey(2023)
    p_values = [0.55, 0.40, 0.25, 0.10, 0.05]
    for p in p_values:
        run(rng, p)