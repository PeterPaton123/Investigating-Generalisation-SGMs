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

## Generate the initial X0 samples from the Gaussian miyture
prior_sample = mixture_prior(jnp.array([1.]), jnp.array([0.]), jnp.array([1.]), num_samples = 2 * 10 ** 3)

"""
P(Xt = xt | X0 = x0)
"""
def p_xt_given_x0(t, xt, x0):
    total = 0.0
    for i in range(np.size(prior_weight)):
        total += prior_weight[i] * pdf_normal(x0 * jnp.exp(-0.5 * (T-t)), var= 1 - np.exp(-(T-t)), x=xt)        
    return total

"""
Expected pdf P(X(t) = yt) at time t
"""
def expected_pdf_from_samples(samples, t, yt):
    partial = Partial(p_xt_given_x0, t, yt)
    return jnp.mean(jax.vmap(partial)(samples))

def u_ou(t, yt):
    pdf_at_time_t = Partial(expected_pdf_from_samples, prior_sample, t)
    return 0.5 * yt + (jax.grad(pdf_at_time_t)(yt))/(pdf_at_time_t(yt))

@jit
def s_ou(t, yt):
    return 1

## Construct the stochastic differential equation
sde_ou = SDE(prior_sample, dt = 1. / 100, u=u_ou, s=s_ou)

## Perform a discretisation of the stochastic differential equation
sde_ou.step_up_to_T(T)

xs_for_pdf = jnp.linspace(-10, 10, num=2000)
ts = sde_ou.ts

pdf_at_time_t = np.zeros((jnp.size(ts), 2000))

for i in range(jnp.size(ts)):
    t = ts[i]
    partial = Partial(expected_pdf_from_samples, prior_sample, t)
    pdf_at_time_t[i, :] = jax.vmap(partial)(xs_for_pdf)

plot(sde_ou.ts, sde_ou.samples, xs_for_pdf, pdf_at_time_t, jnp.size(sde_ou.samples[:, 0]), int (1. / sde_ou.dt), T)
