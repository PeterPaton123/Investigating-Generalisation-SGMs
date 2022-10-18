from copy import copy
from math import sqrt
import numpy as np
import random
import matplotlib.pyplot as plt
import numpy.matlib
from jax import jit
from jax.tree_util import Partial
import jax.numpy as jnp
from matplotlib.colors import LogNorm
    
def pdf_normal(mean, sd, x):
    return 1. / (sd * np.sqrt(2 * np.pi)) * np.exp (- (x - mean) ** 2 / (2 * sd ** 2))

"""
Integral over x0 of P(Xt = xt | X0 = x0)*P(X0 = x0)
dX = -1/2 Xt dt + Bt
With solution Xt = X0 * e^(N(-t, t))
P(Xt = xt | X0 = x0) = P(ln(Xt - X0) = xt - x0) ~ N(-t, t)
"""
def pdf_xt(x0s, pdf_X0s, xt):
    diffs = np.array


def mixture_prior(ws, us, vs, n : int):# -> jnp.array[float]:
    ## Third case checked by transitivity
    assert (np.size(ws) == np.size(us))
    assert (np.size(us) == np.size(vs))
    distributions : int = np.size(ws)
    chosens = jnp.array(np.random.choice(distributions, n, p=ws))
    return jnp.array([ np.random.normal(loc = us[chosen], scale = vs[chosen]) for chosen in chosens])

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.3, 0.7])
prior_means = jnp.array([-3., 7.])
prior_variance = jnp.array([1., 1.])
samples = 10 ** 4
timesteps_per_second = 5 * (10 ** 2)
time_range = 10

prior_sample = mixture_prior(prior_weight, prior_means, prior_variance, samples)

@jit 
def u(t, xt):
    return -0.5 * xt

@jit
def s(xt, t):
    return 1

xs = np.zeros((samples, time_range * timesteps_per_second))
ts = np.linspace(start=0, stop=time_range, num=time_range * timesteps_per_second)
xs[:, 0] = prior_sample
ts[0] = 0

dt = 1. / timesteps_per_second

for i in range(time_range * timesteps_per_second - 1):
    prevXs = xs[:, i]
    t = ts[i]
    uPartial = Partial(u, t)
    sPartial = Partial(s, t)
    rands = np.random.normal(loc = 0, scale = np.sqrt(dt), size=samples)
    xs[:, i+1] = prevXs + dt * uPartial(prevXs) + np.dot(rands, sPartial(prevXs))

fig, axes = plt.subplots(nrows=3, figsize=(6, 8), constrained_layout=True)

axes[0].hist(xs[:, 0], bins=100)

y_fine = xs.flatten()
x_fine = np.matlib.repmat(ts, samples, 1).flatten()

cmap = copy(plt.cm.cividis)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[800, timesteps_per_second])
pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1.5e3), rasterized=True)
fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
axes[1].set_title("2d histogram and log color scale")
axes[1].plot(ts, np.zeros(time_range * timesteps_per_second), color = 'r', linestyle='--')

axes[2].hist(xs[:, -1], bins=100)

plt.show()
