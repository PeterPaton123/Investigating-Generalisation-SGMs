from jax import jit
from jax.tree_util import Partial
import jax.numpy as jnp
from SDE import SDE
from wasserstein_distance import ws_dist_normal
import jax.random as random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/src")
from datasets_and_metrics_pkg import GMM

T = 1.0

def expectation(beta, t):
    return jnp.exp(- beta * t)

def beta_experiment(dt, beta, sample_rng):
    @jit
    def u_ou(t, xt):
        return -beta * xt

    @jit
    def s_ou(t, xt):
        return jnp.sqrt(2.0 * beta)
    
    beta_expectation = Partial(expectation, beta)

    samples = 2 * random.uniform(key=sample_rng, shape=(10_000, ))
    # Construct the stochastic differential equation
    sde_ou = SDE(samples, dt = dt, u=u_ou, s=s_ou)
    # Perform a discretisation of the stochastic differential equation
    sde_ou.step_up_to_T(T)

    ts = np.linspace(0, 1, 100)
    fig, axs = plt.subplots(1)
    axs.plot(ts, beta_expectation(ts), 'k:', label=r'$\mathbb{E}[X_{t}]$')
    axs.plot(sde_ou.ts, jnp.mean(sde_ou.samples, axis=0), 'r', label='$\mathbb{E}[X^{\Delta t}_{t}]$')
    axs.set_xlabel('t')
    axs.legend()
    axs.set_title(r'Expected value and expectation of discretised process, $\beta$='+f'{beta}, dt={dt}')
    axs.grid(True)
    fig.savefig(fname=f'convergence_results/uniform/bin/beta-{beta}-dt-{dt}.png')
    return jnp.abs(expectation(beta, 1) - jnp.mean(sde_ou.samples[:, -1]))

rng = random.PRNGKey(2023)
betas = np.linspace(start=1, stop=20, num=200)
dts = np.linspace(start=0.001, stop=0.1, num=20, endpoint=True)
results = np.zeros((len(dts), len(betas)))
for dt_i, dt in enumerate(dts):
    for beta_i, beta in enumerate(betas):
        rng, sample_rng = random.split(rng, 2)
        results[beta_i, dt_i] = beta_experiment(dt, beta, sample_rng)
        quit()

pd.DataFrame(results).to_csv("convergence_results/uniform/out.csv", index=False, header=False)


