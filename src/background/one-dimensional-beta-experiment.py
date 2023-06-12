from jax import jit
import jax.numpy as jnp
from SDE import SDE
from wasserstein_distance import ws_dist_normal
import jax.random as random
import numpy as np
import pandas as pd

import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/src")
from datasets_and_metrics_pkg import GMM

T = 1.0

def beta_experiment(beta, mu, rng):
    @jit
    def u_ou(t, xt):
        return -beta * xt

    @jit
    def s_ou(t, xt):
        return jnp.sqrt(2.0 * beta)

    mus = jnp.array([[-mu], [mu]])
    covars = jnp.array([jnp.eye(1), jnp.eye(1)])
    gmm = GMM(mus, covars)
    samples = gmm.sample(10_000, sample_rng)

    ## Construct the stochastic differential equation
    sde_ou = SDE(samples, dt = 1. / 100, u=u_ou, s=s_ou)
    ## Perform a discretisation of the stochastic differential equation
    sde_ou.step_up_to_T(T)
    res = ws_dist_normal(sde_ou.samples[:, -1], 0, 1)
    print(f"Mu: {mu}, Beta: {beta}, dist: {res}")
    return res

rng = random.PRNGKey(2023)
betas = np.linspace(start=1, stop=20, num=200)
const_sqrt = np.sqrt(10)
mus = np.array([1, const_sqrt, 10 * const_sqrt, 100, 100 * const_sqrt, 1_000, 1_000 * const_sqrt, 10_000, 10_000 * const_sqrt, 100_000])
results = np.zeros((len(mus), len(betas)))
for mu_i in range(len(mus)):
    for beta_i in range(len(betas)):
        mu = mus[mu_i]
        beta = betas[beta_i]
        rng, sample_rng = random.split(rng, 2)
        results[mu_i, beta_i] = beta_experiment(beta, mu, sample_rng)

pd.DataFrame(results).to_csv("results/one-dim-gauss/out.csv", index=False, header=False)


