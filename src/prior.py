from math import sqrt
import numpy as np
import jax.numpy as jnp

"""
Integral over x0 of P(Xt = xt | X0 = x0)*P(X0 = x0)
dX = -1/2 Xt dt + Bt
With solution Xt = X0 * e^(N(-t, t))
P(Xt = xt | X0 = x0) = P(ln(Xt - X0) = xt - x0) ~ N(-t, t)

def pdf_xt(x0s, pdf_X0s, xt):
    diffs = np.array
"""

def mixture_prior(ws, us, vs, n : int):# -> jnp.array[float]:
    ## Third case checked by transitivity
    assert (np.size(ws) == np.size(us))
    assert (np.size(us) == np.size(vs))
    distributions : int = np.size(ws)
    chosens = jnp.array(np.random.choice(distributions, n, p=ws))
    return jnp.array([ np.random.normal(loc = us[chosen], scale = vs[chosen]) for chosen in chosens])