from multivariate_sde import MV_SDE
from jax import jit
import jax.numpy as jnp
from SDE import SDE
from wasserstein_distance import ws_dist_normal
import jax.random as random
import numpy as np
import pandas as pd

import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/src")
from datasets_and_metrics_pkg import make_union_circle, sliced_wasserstein


T = 1.0

def beta_experiment(beta, mu, test_samples, projection_rng):
    @jit
    def u_ou(t, xt):
        return -beta * xt

    @jit
    def s_ou(t, xt):
        return jnp.sqrt(2.0 * beta)

    samples = make_union_circle(10_000, -mu)
    sde = MV_SDE(samples, dt= 1. / 100, u=u_ou, s=s_ou)
    sde.step_up_to_T(1)
    a = sliced_wasserstein(test_samples, samples, 100, projection_rng)
    print(a)
    return sliced_wasserstein(test_samples, samples, 100, projection_rng)

rng = random.PRNGKey(2023)
rng, test_rng = random.split(rng, 2)
test_set = random.normal(test_rng, shape=(10_000, 2))
betas = np.linspace(start=1, stop=20, num=200)
const_sqrt = np.sqrt(10)
mus = np.array([1, const_sqrt, 10 * const_sqrt, 100, 100 * const_sqrt, 1_000, 1_000 * const_sqrt, 10_000, 10_000 * const_sqrt, 100_000])
results = np.zeros((len(mus), len(betas)))
for mu_i in range(len(mus)):
    for beta_i in range(len(betas)):
        mu = mus[mu_i]
        beta = betas[beta_i]
        rng, projection_rng = random.split(rng, 2)
        results[mu_i, beta_i] = beta_experiment(beta, mu, test_set, projection_rng)

pd.DataFrame(results).to_csv("results/union/out.csv", index=False, header=False)