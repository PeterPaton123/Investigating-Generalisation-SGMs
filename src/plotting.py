import numpy as np
import numpy.matlib
from copy import copy
import matplotlib
import PyQt5
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pdf_utils import pdf_normal

from jax.tree_util import Partial

pdf1 = Partial(pdf_normal, -5., 1.)
pdf2 = Partial(pdf_normal, 5., 1.)

xs_1 = np.linspace(-20, 20, num=200)
p_x0s = 0.5 * pdf1(xs_1) + 0.5 * pdf2(xs_1)  

pdf3 = Partial(pdf_normal, 0., 62)
p_x30s = pdf3(xs_1)

def plot(ts, samples_at_t, xs, expected_pdf_at_time_t, num_samples, timesteps_per_second, time_range):
    fig, axes = plt.subplots(nrows=4, figsize=(6, 8), constrained_layout=True)

    # Prior distrubtion
    axes[0].hist(samples_at_t[:, 0], bins=100, density=True, stacked=True)
    axes[0].plot(xs, expected_pdf_at_time_t[0], color = 'r', linestyle='--')
    y_fine = samples_at_t.flatten()
    x_fine = np.matlib.repmat(ts, num_samples, 1).flatten()

    ## Simulated paths
    cmap = copy(plt.cm.cividis)
    cmap.set_bad(cmap(0))
    h, xedges, yedges = np.histogram2d(x_fine, y_fine, bins=[800, timesteps_per_second])
    pcm = axes[1].pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmax=5e3), rasterized=True)
    fig.colorbar(pcm, ax=axes[1], label="# points", pad=0)
    axes[1].set_title("2d histogram and log color scale")
    axes[1].plot(ts[1:], np.zeros(time_range * timesteps_per_second), color = 'r', linestyle='--')

    ## Expected pdf:
    mesh = axes[2].pcolormesh(ts, xs, np.array(expected_pdf_at_time_t).T, cmap=cmap)
    fig.colorbar(mesh, ax=axes[2], label="Expected pdf in t", pad=0)

    ## pdf at t = T
    axes[3].hist(samples_at_t[:, -1], bins=100, density=True, stacked=True)
    axes[3].plot(xs, expected_pdf_at_time_t[-1], color = 'r', linestyle='--')
    plt.show()
