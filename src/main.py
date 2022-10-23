from jax import jit
import jax.numpy as jnp
from jax.tree_util import Partial
from prior import mixture_prior
from plotting import plot
import numpy as np

## Prior distribution is a mixture of Gaussians:
prior_weight = jnp.array([0.3, 0.7])
prior_means = jnp.array([-3., 7.])
prior_variance = jnp.array([1., 1.])
samples = 2 * 10 ** 4
timesteps_per_second = 5 * (10 ** 2)
time_range = 10

prior_sample = mixture_prior(prior_weight, prior_means, prior_variance, samples)

xs = np.zeros((samples, time_range * timesteps_per_second))
ts = np.linspace(start=0, stop=time_range, num=time_range * timesteps_per_second)
xs[:, 0] = prior_sample
ts[0] = 0

@jit
def u(t, xt):
    return 0 * xt

@jit
def s(t, xt):
    return 1

dt = 1. / timesteps_per_second

for i in range(time_range * timesteps_per_second - 1):
    prevXs = xs[:, i]
    t = ts[i]
    uPartial = Partial(u, t)
    sPartial = Partial(s, t)
    rands = np.random.normal(loc = 0, scale = np.sqrt(dt), size=samples)
    xs[:, i+1] = prevXs + dt * uPartial(prevXs) + sPartial(prevXs) * np.dot(rands, sPartial(prevXs))

def expected_pdf1(p_xt_given_x0, p_x0, x0s, xt, t):
    p_xt_given_x0 = Partial(p_xt_given_x0, xt, t)
    integral = np.dot (p_xt_given_x0(x0s), p_x0(x0s))
    return np.sum(integral) * ((np.max(x0s) - np.min(x0s)) / x0s.size()) 

def pdf_normal(mean, var, x):

    sd = np.sqrt(var)
    return 1. / (sd * np.sqrt(2 * np.pi)) * np.exp (-((x - mean) ** 2 / (2 * var)))

def p_x0(ws, us, vars, x0):
    total = 0
    for i in range(np.size(ws)):
        ws[i] * pdf_normal(us[i], vars[i], x0)
    return total

def p_xt_given_x0(xt, t, x0):
    return pdf_normal(0, t, xt-x0)

p_x02 = Partial(prior_weight, prior_means, prior_variance)
x0s = np.linspace(-15, 15, 3000)
t = np.linspace(0, time_range, timesteps_per_second * time_range)
xts = np.linspace(-15, 15, 3000)

ys = np.zeros(3000, (time_range * timesteps_per_second))
for i in range(3000):
    ys[i] = expected_pdf1(p_xt_given_x0, p_x02, x0s, xt)

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

plot(ts, xs, samples, timesteps_per_second, time_range)