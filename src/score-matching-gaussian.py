from multiprocessing.spawn import import_main_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from copy import copy

N = 100000
M = 300

samples = np.random.normal(loc=0, scale=1, size=N)

def J(mu, sigma):
    return (0.5 * (np.mean(np.square(samples - mu))) /  (sigma**4.0)) - 1 / (sigma**2.0)

Js = np.zeros((M, M))
mus = np.linspace(-2, 2, num=M)
sigmas = np.linspace(0.1, 2.1, num=M)

for i in range(M):
    for j in range(M):
        Js[i, j] = J(mus[i], sigmas[j])

cmap = copy(plt.cm.magma).reversed()
cmap.set_bad(cmap(0))
pcm = plt.pcolormesh(mus, sigmas, Js, cmap=cmap, norm=LogNorm(), rasterized=True)
plt.colorbar(pcm, label="J", pad=0)
plt.show()