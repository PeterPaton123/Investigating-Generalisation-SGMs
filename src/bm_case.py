from abc import abstractmethod
from jax import jit
import jax.numpy as jnp
from prior import mixture_prior
import numpy as np
from jax.tree_util import Partial
from plotting import plot
from pdf_utils import pdf_normal
from SDE import SDE

"""
Numerical solution of the following Stochastic differential equation:
dX = u(X(t), t) dt + s(X(t), t) * dWt

With u and s defined as followed:

"""

T = 30

## This case is defined as dX(t) = dWt

@jit
def u_bm(t, xt):
    return 0

@jit
def s_bm(t, xt):
    return 1

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.5, 0.5])
prior_means = jnp.array([-5., 5.])
prior_variance = jnp.array([1., 1.])

"""
Expected pdf P(X(t) = xt) at time t
"""
def expected_pdf(t, xt):
    total = 0
    for i in range(np.size(prior_weight)):
        total += prior_weight[i] * pdf_normal(mean=prior_means[i], var=prior_variance[i] + t, x=xt)
    return total

## Generate the initial X0 samples from the Gaussian mixture
prior_sample = mixture_prior(prior_weight, prior_means, prior_variance, num_samples = 2 * 10 ** 4)

## Construct the stochastic differential equation
sde_bm = SDE(prior_sample, dt = 1. / 100, u=u_bm, s=s_bm)

## Perform a discretisation of the stochastic differential equation
sde_bm.step_up_to_T(T)

xs_for_pdf = np.linspace(-20, 20, num=2000)
ts = sde_bm.ts

partial_expected_pdf_at_time_t = Partial(expected_pdf, xs_for_pdf)

pdf_at_time_t = np.zeros((np.size(ts), 2000))

for i in range(np.size(ts)):
    t = ts[i]
    partial = Partial(expected_pdf, t)
    pdf_at_time_t[i, :] = partial(xs_for_pdf)

plot(sde_bm.ts, sde_bm.samples, xs_for_pdf, pdf_at_time_t, np.size(sde_bm.samples[:, 0]), int (1. / sde_bm.dt), T)
