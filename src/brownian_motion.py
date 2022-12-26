from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from numpy import matlib
import scipy.stats as stats

T = 1000
N = 10000
INITIAL_PRICE = 100
ts = np.linspace(0, T, num=T+1)

discrete_walk = np.resize(INITIAL_PRICE * np.ones(N), (N, 1))

for i in ts[1:]:
    discrete_walk = np.column_stack((discrete_walk, np.resize(discrete_walk[:, -1], (N, 1)) + np.resize(np.random.normal(loc=0, scale=1, size=N), (N, 1))))

fig, axs = plt.subplots(1, 2)
fig.suptitle('Simple random walks')

for i in range(min(20, N)):
    axs[0].plot(ts, discrete_walk[i, :], linewidth=0.75)

axs[0].set_title("Simple brownian motion paths")
axs[1].set_title("Histogram of brownian motion paths")
axs[0].plot(ts, 100 * np.ones(T+1), color='k', linewidth=1.5, linestyle='--')
axs[1].hist(discrete_walk[:, -1], density=True, stacked=True, bins=25)
axs[1].plot(np.linspace(100 - 4 * np.sqrt(T), 100 + 4 * np.sqrt(T), 1000), stats.norm.pdf(np.linspace(100 - 4 * np.sqrt(T), 100 + 4 * np.sqrt(T), 1000), loc=100, scale=np.sqrt(T)), color='r', linewidth=1.5, linestyle='-')

plt.show()


"""
Histograms in time, not so useful due t discretisation of data
y_fine = discrete_walk.flatten()
x_fine = matlib.repmat(ts, N, 1).flatten()

cmap = copy(plt.cm.cividis)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[20, T])
pcm = axs[2].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1e2), rasterized=True)
fig.colorbar(pcm, ax=axs[2], label="# points", pad=0)
axs[2].set_title("Histogram in time for discrete case")

y_fine = discrete_walk.flatten()
x_fine = matlib.repmat(ts, N, 1).flatten()

cmap = copy(plt.cm.cividis)
cmap.set_bad(cmap(0))
h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[20, T])
pcm = axs[2].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1e2), rasterized=True)
fig.colorbar(pcm, ax=axs[2], label="# points", pad=0)
axs[2].set_title("Histogram in time for discrete case")
"""

