from jax import jit
import jax.numpy as jnp
from jax.tree_util import Partial
from prior import mixture_prior, mixture_prior_pdf
from pdf_utils import pdf_normal, expected_pdf_at_time_t
from plotting import plot
import numpy as np

"""
Numerical solution of the following Stochastic differential equation:
dX = u(X(t), t) dt + s(X(t), t) * dWt

With u and s defined as followed:

"""

## This case is defined as dX(t) = dWt

@jit
def u(t, xt):
    return -0.5 * xt

@jit
def s(t, xt):
    return 1

"""
t : at time t
xt : A single xt
x0s : An array of possible X0 values 
Returns: An array of P(X(t) = xt | X0 = x0) for all X0s entered
"""
def p_xt_given_x0s(t, x0s, xt):
    partial_pdf_brownian = Partial(pdf_normal, 0, t-0)
    diffs = -1 * (x0s - xt)
    return partial_pdf_brownian(diffs)

## Number of samples
num_samples = 2 * 10 ** 4

## Sample rate and range
timesteps_per_second = 5 * (10 ** 2)
time_range = 10

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.3, 0.7])
prior_means = jnp.array([-5., 5.])
prior_variance = jnp.array([1., 1.])

## Generate the initial X0 samples from the Gaussian mixture
prior_sample = mixture_prior(prior_weight, prior_means, prior_variance, num_samples)

## Xt values to consider in plotting pdf
xs = np.linspace(-10, 10, num=200)

## Expected distribution of the initial samples (X0)
prior_pdf_partial = Partial(mixture_prior_pdf, ws=prior_weight, us=prior_means, vars=prior_variance)
prior_pdf = prior_pdf_partial(x0=xs)

## Perform the discretisation of the forward time step
samples_at_t = np.zeros((num_samples, time_range * timesteps_per_second))
ts = np.linspace(start=0, stop=time_range, num=time_range * timesteps_per_second)
samples_at_t[:, 0] = prior_sample
ts[0] = 0

dt = 1. / timesteps_per_second

for i in range(time_range * timesteps_per_second - 1):
    prevXs = samples_at_t[:, i]
    t = ts[i]
    uPartial = Partial(u, t)
    sPartial = Partial(s, t)
    rands = np.random.normal(loc = 0, scale = np.sqrt(dt), size=num_samples)
    samples_at_t[:, i+1] = prevXs + dt * uPartial(prevXs) + sPartial(prevXs) * np.dot(rands, sPartial(prevXs))

partial_expected_pdf_at_time_t = Partial(expected_pdf_at_time_t, p_xt_given_x0s, prior_pdf, xs)

expected_pdf_at_time_t_2 = np.array([partial_expected_pdf_at_time_t(t) for t in ts[1:]])

plot(ts, samples_at_t, xs, expected_pdf_at_time_t_2, num_samples, timesteps_per_second, time_range)