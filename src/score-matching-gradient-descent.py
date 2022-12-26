import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.tree_util import Partial

N = 100
samples = np.resize(np.random.normal(loc = 0, scale=1, size=2 * N), (N, 2))

def f1(v, mu_1, cov_1):
    return -(0.5) * (v - mu_1) @ cov_1 @ (v - mu_1).T

def f2(v, mu_2, cov_2):
    return -(0.5) * (v - mu_2).T @ cov_2 @ (v - mu_2)


def score_fun(theta, x):
    mu_1 = jnp.resize(theta[:1], (2, ))
    cov_1 = jnp.resize(theta[2:6], (2, 2))
    mu_2 = jnp.resize(theta[7:8], (2, ))
    cov_2 = jnp.resize(theta[9:], (2, 2))
    return jnp.log(jnp.exp(f1(x, mu_1, cov_1)) + jnp.exp(f2(x, mu_2, cov_2)))

def J(theta):
    score_partial = Partial(score_fun, theta)
    jacobian = jacfwd(score_partial)
    hessian = jacfwd(jacrev(score_partial))
    total = 0
    for i in range(N):
        x = jnp.reshape(samples[i], (2, ))
        total += hessian(x)[0, 0] + hessian(x)[1, 1] + 0.5 * ((jacobian(x)[0])**2 + (jacobian(x)[1])**2) 
    return total

steps = 100
theta = jnp.array(np.random.uniform(0, 5, size=12))
learning_rate = 0.1

for i in range(steps):
    print(i)
    theta -= learning_rate * jacfwd(J)(theta)

print(theta)

