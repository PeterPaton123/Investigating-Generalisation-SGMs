import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Logpathsorm
from numpy import matlib
import scipy.stats as stats
from jax.tree_util import Partial

T = 10
dt = 0.1
paths = 10
IpathsITIAL_PRICE = 100
ts = np.linspace(0, T, num=T/dt + 1)

def f(t, x_t):
    return x_t + t

def g(x_t, t):
    return 1

ito_walk = np.resize(IpathsITIAL_PRICE * np.ones(paths), (paths, 1))
strat_walk = np.resize(IpathsITIAL_PRICE * np.ones(paths), (paths, 1))

for t in ts[1:]:
    partial_f = Partial(f, t)
    prev_xs = np.resize(ito_walk[:, -1], (paths, 1))
    ito_walk = np.column_stack((ito_walk, prev_xs) + partial_f(prev_xs) * dt + np.resize(np.random.normal(loc=0, scale=dt, size=paths), (paths, 1))))

fig, axs = plt.subplots(1, 2)
fig.suptitle('Simple random walks')