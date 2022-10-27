from abc import abstractmethod
from jax import jit
import jax.numpy as jnp
from prior import mixture_prior
import numpy as np
from jax.tree_util import Partial
from plotting import plot
from pdf_utils import pdf_normal
from SDE import SDE

"""z
Numerical solution of the following Stochastic differential equation:
dX = u(X(t), t) dt + s(X(t), t) * dWt

With u and s defined as followed:

"""

T = 10

## This case is defined as dX(t) = -0.5dt + dWt

@jit
def u_ou(t, xt):
    return -0.5 * xt

@jit
def s_ou(t, xt):
    return 1

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.5, 0.5])
prior_means = jnp.array([-5., 5.])
prior_variance = jnp.array([1., 1.])

"""
Expected pdf P(X(t) = xt) at time t
"""
def expected_pdf(t, xt):
    total = 0.0
    for i in range(np.size(prior_weight)):
        total += prior_weight[i] * pdf_normal(mean=prior_means[i] * np.exp(-0.5 * t), var=(prior_variance[i] - 1.) * np.exp(-t) + 1., x=xt)
    return total

## Generate the initial X0 samples from the Gaussian mixture
prior_sample = mixture_prior(prior_weight, prior_means, prior_variance, num_samples = 2 * 10 ** 4)

## Construct the stochastic differential equation
sde_ou = SDE(prior_sample, dt = 1. / 100, u=u_ou, s=s_ou)

## Perform a discretisation of the stochastic differential equation
sde_ou.step_up_to_T(T)

xs_for_pdf = np.linspace(-10, 10, num=2000)
ts = sde_ou.ts

partial_expected_pdf_at_time_t = Partial(expected_pdf, xs_for_pdf)

pdf_at_time_t = np.zeros((np.size(ts), 2000))

for i in range(np.size(ts)):
    t = ts[i]
    partial = Partial(expected_pdf, t)
    pdf_at_time_t[i, :] = partial(xs_for_pdf)

plot(sde_ou.ts, sde_ou.samples, xs_for_pdf, pdf_at_time_t, np.size(sde_ou.samples[:, 0]), int (1. / sde_ou.dt), T)