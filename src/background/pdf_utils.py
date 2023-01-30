import jax.numpy as jnp
import numpy as np
from jax.tree_util import Partial

"""
Calculates the probability X(t) = x for each of the xs in the input array
Input:
xs : Range of x values
p_x0s: Prior distribution of xs
p_xt_given_x0: calculates P(X(t) = xt | X0 = x0) for all x0s in xs
t : time

Returns: at a time T an array of P(X(t) = xt) for each xt in xs
"""

def expected_pdf_at_time_t(p_xt_given_x0s, p_x0s, xs, t):
    x_interval = (jnp.max(xs) - jnp.min(xs)) / np.size(xs)
    p_xt_given_x0s_partial = Partial(p_xt_given_x0s, t, xs)
    # An array of each Xt and the P(Xt = xt | X0 = x0), for all x0s at time t
    p_xts_given_x0s = jnp.array([p_xt_given_x0s_partial(xt) for xt in xs])
    return np.array([ x_interval * jnp.dot(p_x0s, p_xt_given_x0s) for p_xt_given_x0s in p_xts_given_x0s])

"""
Probability density at a point X from a normal distribution mean mean, variance var
"""
def pdf_normal(mean, var, x):
    sd = jnp.sqrt(var)
    return jnp.exp (-((x - mean) ** 2 / (2 * var))) / (sd * jnp.sqrt(2 * jnp.pi))

def pdf_log_normal(mean, var, x):
    sd = jnp.sqrt(var)
    return 1. / (jnp.log(x) * sd * jnp.sqrt(2 * jnp.pi)) * jnp.exp (-((jnp.log(x) - mean) ** 2 / (2 * var)))