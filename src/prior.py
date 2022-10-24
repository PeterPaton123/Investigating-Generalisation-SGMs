from math import sqrt
from pdf_utils import pdf_normal
import numpy as np
import jax.numpy as jnp

"""
Inputs
ws weights
us means
vs variances
n number of samples to generate

Returns: n samples from the gaussian mixture defined by inputs
"""

def mixture_prior(ws, us, vs, num_samples : int):
    ## Third case checked by transitivity
    assert (np.size(ws) == np.size(us))
    assert (np.size(us) == np.size(vs))
    distributions : int = np.size(ws)
    chosens = jnp.array(np.random.choice(distributions, num_samples, p=ws))
    return jnp.array([ np.random.normal(loc = us[chosen], scale = vs[chosen]) for chosen in chosens])

"""
Inputs
ws weights
us means
vs variances
X0: pdf at this point

Returns: pdf of the gaussian mixture at the point x0
"""
def mixture_prior_pdf(ws, us, vars, x0):
    total = 0
    for i in range(np.size(ws)):
        total += ws[i] * pdf_normal(us[i], vars[i], x0)
    return total