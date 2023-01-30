import scipy
import numpy as np
import jax
import matplotlib.pyplot as plt
import jax.numpy as jnp

def ws_dist_two_samples(sample1, sample2):
    """Wasserstein distance between two samples."""
    return scipy.stats.wasserstein_distance(sample1, sample2, u_weights=None, v_weights=None)

def ws_dist_sample_pdf(sample, pdf):
    return ws_dist_two_samples(sample, np.random.choice(sample, size=jnp.size(sample), p=pdf/np.sum(pdf)))

def ws_dist_normal(samples, mean, var): 
    """Wasserstein distance to standard normal"""
    n = np.shape(samples)[0]
    ## Calculates the inverse cdf of the standard normal distribution
    true_inv_cdf = jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, n)[1:-1], loc=mean, scale=np.sqrt(var))
    approx_inverse_cdf = jnp.sort(samples)[1:-1]
    return np.mean(jnp.abs(true_inv_cdf - approx_inverse_cdf))

def ws_dist_normal_mixture(samples, ws, ms, vs):
    n = np.shape(samples)[0]
    true_inv_cdf = 0
    for i in range(np.size(ws)):
        sd = np.sqrt(vs[i])
        true_inv_cdf += ws[i] * jax.scipy.stats.norm.ppf(jnp.linspace(0, 1, n)[1:-1], loc=ms[i], scale=sd)
    approx_inverse_cdf = jnp.sort(samples)[1:-1]
    a = np.mean(jnp.abs(true_inv_cdf - approx_inverse_cdf))
    if (not np.isfinite(a)):
        print("ERROROR 2")
    return a

dists = np.zeros(100)

sample1 = np.linspace(-5, 5, 1000)

for i in range(100):
    mu = -2.5 + (i / 20)
    sample2 = np.linspace(-3 + mu, mu + 3, 1000)
    dists[i] = scipy.stats.wasserstein_distance(sample1, sample2, u_weights=scipy.stats.norm(loc=0, scale=1).pdf(sample1), v_weights=scipy.stats.norm(loc=mu, scale=1).pdf(sample2))

fig, axs = plt.subplots(1, 2)
fig.suptitle('Wasserstein distance between moving normal distributions')

y1=scipy.stats.norm(loc=0, scale=1).pdf(sample1)
y2=scipy.stats.norm(loc=-1, scale=1).pdf(sample1)
axs[0].fill_between(sample1, y1, 0, where=y1>0, color='b', alpha=0.5, label='Fixed distribution')
axs[0].fill_between(sample1, y2, 0, where=y2>0, color='r', alpha=0.5, label='Moving mean')
axs[0].plot([-2.5, -2.5], [0, 0.4], linewidth=1.5, linestyle='--', color='k')
axs[0].plot([2.5, 2.5], [0, 0.4], linewidth=1.5, linestyle='--', color='k')
axs[0].scatter([-1], [scipy.stats.norm.pdf(loc=-1, scale=1, x=-1)], color='r')
axs[0].legend(loc='upper right')
axs[1].plot(np.linspace(-2.5, 2.45, num=100), dists)
axs[1].plot([-2.5, -2.5], [0, 2.5], linewidth=1.5, linestyle='--', color='k')
axs[1].plot([2.5, 2.5], [0, 2.5], linewidth=1.5, linestyle='--', color='k')

plt.show()
