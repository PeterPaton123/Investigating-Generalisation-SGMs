from inspect import stack
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap
from jax.tree_util import Partial

from InteractingSDE import InteractingSDE
from pdf_utils import pdf_normal
from plotting import plot
from prior import mixture_prior
from wasserstein_distance import ws_dist_normal

N = 1000
dt = 1 / 100
T = 5

"""
Interacting SDE of the form:
dXn(t) = -0.5Xn(t) - sum (1 / n) (Xi(t) - Xj(t)) + dW(t)
"""

A = np.ones((N, N))
A = A / N
alpha = 0.5

for i in range(N):
    A[i, i] = - alpha - 1 + (1 / N)

def A2(x, t):
    return jnp.zeros((N, 1))

B = np.identity(N)

"""
Using this value of A we can show that sigma t has a closed form
"""

"""
Take our X0s from the multivariate normal distribution like before
"""

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.5, 0.5])
prior_means = jnp.array([-5., 5.])
prior_variance = jnp.array([1., 1.])

x0s = mixture_prior(prior_weight, prior_means, prior_variance, N)
x0s = jnp.resize(x0s, (N, 1))

sde = InteractingSDE(x0s, dt, A, A2, B)
sde.step_up_to_T(T)

xs_for_pdf = np.linspace(-10, 10, num=2000)
ts = sde.ts

print(sde.samples[:, 50])
plt.plot(np.linspace(-3, 3, 30), vmap(lambda x: pdf_normal(0,1, x))(np.linspace(-3, 3, 30)), linestyle='-', color='r')
plt.show()
