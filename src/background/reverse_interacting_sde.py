import numpy as np
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal, norm
import scipy.stats
import matplotlib.pyplot as plt

from jax.tree_util import Partial
from jax import jacfwd, vmap, jit
from jax.scipy.special import logsumexp
from functools import partial

from InteractingSDE import InteractingSDE

T = 5
N = 10
dt = 1 / 20

A = np.ones((N, N))
A = A / N
alpha = 0.5

for i in range(N):
    A[i, i] = - alpha - 1 + (1 / N)

A = -A
B = np.identity(N)

def e_theta_not_diag(t):
    return np.exp(- alpha * t) * (1 - np.exp(-t)) / N

def e_theta_diag(t):
    return np.exp(- alpha * t) * ((N - 1) * np.exp(-t) + 1) / N

def e_theta_t_mat(t):
    e_theta = np.empty((N, N))
    e_theta.fill(e_theta_not_diag(t))
    np.fill_diagonal(e_theta, e_theta_diag(t))
    return e_theta

#print("e_theta_mat_shape", np.shape(e_theta_t_mat(0)))

def sigma_diag(t):
    return (1 / N) * (
        (-(N - 1)/(2 + 2 * alpha)) * np.exp(-2 * t * (1 + alpha))
        - (1 / (2 * alpha)) * np.exp(-2 * t * alpha)
        + ((N - 1) / (2 * (1 + alpha))) + (1 / (2 * alpha))
    )

def sigma_not_diag(t):
    return (1 / N) * (
        ((1 / (2 + 2 * alpha)) * np.exp(-2 * t * (alpha + 1)))
        - (1 / (2 * alpha)) * np.exp(-2 * t * alpha)
        + 1 / (2 * alpha * (alpha + 1))
    )

def sigma_t(i, j, t):
    if (i == j):
        return sigma_diag(t)
    return sigma_not_diag(t)

def sigma_t_mat(t):
    sigma = np.empty((N, N))
    not_diag = sigma_not_diag(t)
    diag = sigma_diag(t)
    sigma.fill(not_diag)
    np.fill_diagonal(sigma, diag)
    return jnp.array(sigma)

#print("sigma_theta_mat_shape", np.shape(sigma_t_mat(0)))

prior_means_1 = np.zeros((N, 1))
prior_means_1.fill(-5)
prior_means_2 = np.empty((N, 1))
prior_means_2.fill(5)


# Dynamic programming; storing (functions which give) pdf at x at time t
distributions_at_time_t = {}

def p_t(t, x):
    if not (t in distributions_at_time_t):
        cot_mat_at_t = e_theta_t_mat(2 * t) + sigma_t_mat(t)
        distributions_at_time_t[t] = (lambda x: multivariate_normal.pdf(x, mean = np.resize(e_theta_t_mat(t)@prior_means_1, (N, )), cov=cot_mat_at_t), lambda x: multivariate_normal.pdf(x, mean = np.resize(e_theta_t_mat(t)@prior_means_2, (N, )), cov=cot_mat_at_t))
    (dist_1, dist_2) = distributions_at_time_t[t]
    return 0.5 * dist_1(x) + 0.5 * dist_2(x)
    
    # cot_mat_at_t = e_theta_t_mat(2 * t) + sigma_t_mat(t)
    # return 0.5 * multivariate_normal.pdf(x, mean = np.resize(e_theta_t_mat(t)@prior_means_1, (N, )), cov=cot_mat_at_t) \
    #      + 0.5 * multivariate_normal.pdf(x, mean = np.resize(e_theta_t_mat(t)@prior_means_2, (N, )), cov=cot_mat_at_t)

"""
def A2(x, t):
    p_t_at_T_minus_t = Partial(p_t, T-t)
    def log_p_t_at_t(x2):
        return jnp.log(p_t_at_T_minus_t(x2))
    a = jacfwd(log_p_t_at_t)(jnp.resize(x, (N, )))
    if (not np.all(np.isfinite(a))):
        print("not finite")
    print(a)
    return a
"""

a = 1
print(jnp.shape(a))

def exp_term(mu, sigma, x):
    mu = jnp.resize(mu, (N, 1))
    sigma = jnp.resize(sigma, (N, N))
    x = jnp.resize(x, (N, 1))
    return (-((x- mu).T)@sigma@(x-mu))[0][0]

# Functions defined in overleaf
Vis_at_time_t = {}

def A2(x, s):
    t = T - s
    def func(x2):
        if not (t in Vis_at_time_t):
            inv_cov_mat_at_t = np.linalg.inv(e_theta_t_mat(2 * t) + sigma_t_mat(t))
            e_theta_t = e_theta_t_mat(t)
            V1_at_t = Partial(exp_term, e_theta_t@prior_means_1, inv_cov_mat_at_t)
            V2_at_t = Partial(exp_term, e_theta_t@prior_means_2, inv_cov_mat_at_t)
            Vis_at_time_t[t] = (V1_at_t, V2_at_t)
        Vis = Vis_at_time_t[t]
        Vs = jnp.array([Vis[0](x2), Vis[1](x2)])
        #print(Vs)
        #Vs -= jnp.max(Vs)
        #print(Vs)
        return logsumexp(Vs) #jnp.max(Vs) + logsumexp(Vs)
    a = jacfwd(func)(x)
    p_t_at_T_minus_t = Partial(p_t, t)
    def log_p_t_at_t(x2):
        return jnp.log(p_t_at_T_minus_t(x2))
    b = jacfwd(log_p_t_at_t)(jnp.resize(x, (N, )))
    if (not np.all(np.isfinite(a))):
        print("not finite")
    return b


RUN_FOR = 100
all_samples = np.zeros(RUN_FOR * N)

for i in range(RUN_FOR):
    x0s = scipy.stats.multivariate_normal.rvs(mean = np.zeros(N), cov = e_theta_t_mat(2 * T) + sigma_t_mat(T), size=1)
    x0s = jnp.resize(x0s, (N, 1))
    sde = InteractingSDE(x0s, dt, A, A2, B)
    sde.step_up_to_T(T)
    all_samples[i*N:i*N+N] = np.reshape(sde.samples[:, -1], (N, ))

#plt.plot(sde.samples[:, -1], sde.samples[:, -1], density=True)
#plt.show()

print(all_samples)

plt.hist(all_samples, density=True)
plt.show()
