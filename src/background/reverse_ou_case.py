from abc import abstractmethod
from jax import jit
import jax.numpy as jnp
from prior import mixture_prior
import numpy as np
import jax
from jax.tree_util import Partial
from plotting import plot
from pdf_utils import pdf_normal
from SDE import SDE
from wasserstein_distance import ws_dist_two_samples, ws_dist_normal_mixture
import matplotlib.pyplot as plt

"""
Numerical solution of the following Stochastic differential equation:
dX = u(X(t), t) dt + s(X(t), t) * dWt

With u and s defined as followed:

"""

T = 10

## Prior distribution is a miyture of Gaussians:
prior_weight = jnp.array([0.5, 0.5])
prior_means = jnp.array([-5., 5.])
prior_variance = jnp.array([1., 1.])

"""
Expected pdf P(X(t) = yt) at time t
"""
@jit
def expected_pdf(t, yt):
    total = 0.0
    for i in range(np.size(prior_weight)):
        total += prior_weight[i] * pdf_normal(mean=prior_means[i] * jnp.exp(-0.5 * (T-t)), var=(prior_variance[i] - 1.) * jnp.exp(-(T-t)) + 1., x=yt)
    return total

@jit
def u_ou(t, yt):
    pdf_at_time_t = Partial(expected_pdf, t)
    return 0.5 * yt + (jax.grad(pdf_at_time_t)(yt))/(pdf_at_time_t(yt))

@jit
def s_ou(t, yt):
    return 1

## Generate the initial X0 samples from the Gaussian miyture
prior_sample = mixture_prior(jnp.array([1.]), jnp.array([0.]), jnp.array([1.]), num_samples = 2 * 10 ** 4)

## Construct the stochastic differential equation
sde_ou = SDE(prior_sample, dt = 1. / 100, u=u_ou, s=s_ou)

## Perform a discretisation of the stochastic differential equation
sde_ou.step_up_to_T(T)

xs_for_pdf = jnp.linspace(-10, 10, num=2000)
ts = sde_ou.ts

pdf_at_time_t = np.zeros((jnp.size(ts), 2000))

for i in range(jnp.size(ts)):
    t = ts[i]
    partial = Partial(expected_pdf, t)
    pdf_at_time_t[i, :] = jax.vmap(partial)(xs_for_pdf)

plot(sde_ou.ts, sde_ou.samples, xs_for_pdf, pdf_at_time_t, jnp.size(sde_ou.samples[:, 0]), int (1. / sde_ou.dt), T)

wasserstein_distance_num = int(jnp.shape(sde_ou.ts)[0] / 10)

wasserstein_distance_at_t = np.zeros((wasserstein_distance_num, 2))
for i in range(0, jnp.shape(sde_ou.ts)[0]-1, 10):
    wasserstein_distance_at_t[int(i / 10)] = np.array([sde_ou.ts[i], ws_dist_normal_mixture(sde_ou.samples[:, i], ws=prior_weight, ms=prior_means, vs=prior_variance)])

print(wasserstein_distance_at_t)

plt.plot(wasserstein_distance_at_t.T[0], np.flip(wasserstein_distance_at_t.T[1]), color='r', linestyle='--')
plt.show()