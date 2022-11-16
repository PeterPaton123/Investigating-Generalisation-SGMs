from jax import jit, vmap, grad
import jax.numpy as jnp
from prior import mixture_prior
import numpy as np
from jax.tree_util import Partial
from plotting import plot
from pdf_utils import pdf_normal, pdf_log_normal
import matplotlib.pyplot as plt
from SDE import SDE

"""
Numerical solution of the following Stochastic differential equation:
dX = u(X(t), t) dt + s(X(t), t) * dWt

With u and s defined as followed:

"""

T = 10

## This case is defined as dX(t) = -0.5xdt + xdWt

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([1])
prior_means = jnp.array([5])
prior_variance = jnp.array([1.])

xs_for_pdf = jnp.linspace(-10, 10, num=2000)

def conditional_pdf(t, xt, x0):
    """P(Xt = xt | X0 = x0)"""
    ## Xt has a log normal distribution
    return pdf_log_normal(mean = jnp.log(x0) - t, var = jnp.sqrt(t), x=xt)

def expected_pdf_x0(t, xt, x0):
    total = 0.0
    for i in range(np.size(prior_weight)):
        total += prior_weight[i] * pdf_normal(mean=prior_means[i], var=prior_variance[i], x=x0) * conditional_pdf(t, xt, x0)
    return total

"""
Expected pdf P(X(t) = xt) at time t
"""
def expected_pdf(t, xt):
    p = Partial(expected_pdf_x0, t, xt)
    probabilities = vmap(p)(xs_for_pdf)
    return jnp.sum(probabilities[jnp.isfinite(probabilities)])

@jit
def u_ou(t, yt):
    pdf_t = Partial(expected_pdf, t)
    def sigma_squared_pdf(yt_2):
        return (yt_2 ** 2) * pdf_t(yt_2)
    return 0.5 * yt + 1 / pdf_t(yt) * (sigma_squared_pdf)(yt)

@jit
def s_ou(t, xt):
    return xt

## Generate the initial X0 samples from the Gaussian mixture
prior_sample = mixture_prior(jnp.array([1.]), jnp.array([0.]), jnp.array([0.05]), num_samples = 2 * 10 ** 4)

## Construct the stochastic differential equation
sde_gbm = SDE(prior_sample, dt = 1. / 20, u=u_ou, s=s_ou)

## Perform a discretisation of the stochastic differential equation
sde_gbm.step_up_to_T(T)


xs_for_pdf = jnp.linspace(-10, 10, num=2000)
ts = sde_gbm.ts

pdf_at_time_t = np.zeros((jnp.size(ts), 2000))

partial_expected_pdf_at_time_t = Partial(expected_pdf, xs_for_pdf)

"""
for i in range(jnp.size(ts)):
    t = ts[i]
    ## Time reversal case so t = T - t
    partial = Partial(partial_expected_pdf_at_time_t, T - t)
    pdf_at_time_t[i, :] = vmap(partial)(xs_for_pdf)
"""

if (jnp.any(jnp.isfinite(sde_gbm.samples.flatten()))):
    print("Hello")

plot(sde_gbm.ts, sde_gbm.samples, xs_for_pdf, pdf_at_time_t, jnp.size(sde_gbm.samples[:, 0]), int (1. / sde_gbm.dt), T)
