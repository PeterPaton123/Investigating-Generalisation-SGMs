from re import X
import jax.numpy as jnp
from jax import jacfwd
import numpy as np
from jax.tree_util import Partial
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt

DIMS = 2
delta = 1

prior_mean_1 = np.array([-1, -1])
prior_mean_2 = np.array([ 1,  1])
cov_mat_1 = 0.2*np.eye(2)
cov_mat_2 = 0.2*np.eye(2)

def exp_term(mu, sigma, x):
    mu = jnp.resize(mu, (DIMS, 1))
    sigma = jnp.resize(sigma, (DIMS, DIMS))
    x = jnp.resize(x, (DIMS, 1))
    return (-((x- mu).T)@sigma@(x-mu))[0][0]

v_1 = Partial(exp_term, prior_mean_1, np.linalg.inv(cov_mat_1))
v_2 = Partial(exp_term, prior_mean_2, np.linalg.inv(cov_mat_2))

def log_pdf(x):
    vs = jnp.array([v_1(x), v_2(x)])
    vs -= jnp.max(vs)
    return jnp.max(vs) + logsumexp(vs)

rands = np.random.uniform(-2, 2, size=(10, DIMS))

for i in range(1000):
    for j in range(10):
        rands[j] = rands[j] + 0.1 * jacfwd(log_pdf)(rands[j]) * np.sqrt(2 * delta) * np.random.multivariate_normal(mean=np.array([0., 0.]), cov=np.eye(2))
    if (i % 20 == 0):
        plt.clf()
        plt.scatter(rands[:, 0], rands[:, 1], marker='o', color='r')
        plt.scatter([-1, 1], [-1, 1], marker='x', color='b')
        plt.title(i)
        plt.show(block=False)
        plt.pause(0.01)