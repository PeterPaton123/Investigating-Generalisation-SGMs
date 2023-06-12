from jax import jit, vmap
import jax.numpy as jnp
from prior import mixture_prior
from SDE import SDE
from wasserstein_distance import ws_dist_normal
import jax.random as random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/src")

T = 1.0
beta = 10
rng = random.PRNGKey(2023)
rng, sample_rng = random.split(rng)

@jit
def u_ou(t, xt):
    return -beta * xt

@jit
def s_ou(t, xt):
    return jnp.sqrt(2.0 * beta)

@jit
def cauchy_inverse_cdf(x):
    return (10_000.0 / jnp.pi) * jnp.tan((x - 0.5) * jnp.pi) + 1000

if __name__ == "__main__":
    u_samples = random.uniform(rng, shape=(20_000, ))
    samples = cauchy_inverse_cdf(u_samples)
    results = np.zeros(101)
    sde_ou = SDE(samples, dt = 1. / 100, u=u_ou, s=s_ou)
    ## Perform a discretisation of the stochastic differential equation
    sde_ou.step_up_to_T(T)
    res = np.zeros(101)
    for i in range(101):
        res[i] = ws_dist_normal(sde_ou.samples[:, i], 0, 1)
    plt.plot(range(101), res)
    plt.show()
    print(np.min(res))
