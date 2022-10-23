import numpy as np
import numpy.matlib
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot(ts, xs, samples, timesteps_per_second, time_range):
    fig, axes = plt.subplots(nrows=3, figsize=(6, 8), constrained_layout=True)

    axes[0].hist(xs[:, 0], bins=100, density=True, stacked=True)

    y_fine = xs.flatten()
    x_fine = np.matlib.repmat(ts, samples, 1).flatten()

    cmap = copy(plt.cm.cividis)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[800, timesteps_per_second])
    pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=1.5e3), rasterized=True)
    fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
    axes[1].set_title("2d histogram and log color scale")
    axes[1].plot(ts, np.zeros(time_range * timesteps_per_second), color = 'r', linestyle='--')

    axes[2].hist(xs[:, -1], bins=100, density=True, stacked=True)

    plt.show()
