from multiprocessing.spawn import import_main_path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from copy import copy
from scipy.optimize import minimize

N = 100000
M = 101

samples = np.random.normal(loc=0, scale=1, size=N)

def J(input):
    mu = input[0]
    sigma = input[1]
    return (0.5 * (np.mean(np.square(samples - mu))) /  (sigma**4.0)) - 1 / (sigma**2.0)

Js = np.zeros((M, M))
mus = np.linspace(-2, 2, num=M)
print(mus)
sigmas = np.linspace(0.1, 2.1, num=M)
"""
res = minimize(J, np.array([2., 2.,]), method='BFGS', options={'disp': True})
print(res.x)
"""

for i in range(M):
    for j in range(M):
        Js[i, j] = J([mus[i], sigmas[j]]) - J([0, 1]) + 1

cmap = copy(plt.cm.magma).reversed()
#cmap.set_bad(cmap(0))
pcm = plt.pcolormesh(mus, sigmas, Js[:-1, :-1], cmap=cmap, norm=LogNorm(vmax=1.6), rasterized=True, shading="flat")
plt.xlabel("mu")
plt.ylabel("sigma")
plt.colorbar(pcm, label="J")
plt.show()
