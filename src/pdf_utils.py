import numpy as np
from copy import copy
import jax.numpy as jnp
from jax.tree_util import Partial
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt

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
    x_interval = (np.max(xs) - np.min(xs)) / np.size(xs)
    p_xt_given_x0s_partial = Partial(p_xt_given_x0s, t, xs)
    # An array of each Xt and the P(Xt = xt | X0 = x0), for all x0s at time t
    p_xts_given_x0s = np.array([p_xt_given_x0s_partial(xt) for xt in xs])
    return np.array([ x_interval * np.dot(p_x0s, p_xt_given_x0s) for p_xt_given_x0s in p_xts_given_x0s])

"""
Probability density at a point X from a normal distribution mean mean, variance var
"""
def pdf_normal(mean, var, x):
    sd = np.sqrt(var)
    return 1. / (sd * np.sqrt(2 * np.pi)) * np.exp (-((x - mean) ** 2 / (2 * var)))
