from jax import jacfwd
from jax.scipy.stats import multivariate_normal
import jax.numpy as jnp
import numpy as np

def a(x):
    N = len(x)
    b = lambda y: multivariate_normal.pdf(y, np.zeros(N), np.eye(N))
    return b(x) + b(x)

def b(x):
    return jnp.log(a(x))

print(jacfwd(b)(np.resize(jnp.ones(10), (10, 1))))