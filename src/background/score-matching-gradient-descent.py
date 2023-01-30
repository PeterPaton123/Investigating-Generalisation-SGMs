from re import X
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev
from jax.tree_util import Partial
from jax.scipy.special import logsumexp
from scipy.optimize import minimize

N = 100
samples = np.zeros((N, 2))
for i in range(N):
    if np.random.uniform(low=0, high=1) < 0.5:
        samples[i] = np.reshape(np.random.normal(loc = -5, scale=1, size=2), (2, ))
    else:
        samples[i] = np.reshape(np.random.normal(loc = 5, scale=1, size=2), (2, ))

print(samples)

def f1(v, mu_1, cov_1):
    return -(0.5) * (v - mu_1).T @ cov_1 @ (v - mu_1)

def f2(v, mu_2, cov_2):
    return -(0.5) * (v - mu_2).T @ cov_2 @ (v - mu_2)


def score_fun(theta, x):
    mu_1 = jnp.resize(theta[:1], (2, ))
    cov_1 = jnp.resize(theta[2:6], (2, 2))
    mu_2 = jnp.resize(theta[7:8], (2, ))
    cov_2 = jnp.resize(theta[9:], (2, 2))
    return logsumexp(jnp.array([f1(x, mu_1, cov_1), f2(x, mu_2, cov_2)]))

def J(theta):
    score_partial = Partial(score_fun, theta)
    jacobian = jacfwd(score_partial)
    hessian = jacfwd(jacrev(score_partial))
    total = 0
    for i in range(N):
        x = jnp.reshape(samples[i], (2, ))
        total += (hessian(x)[0, 0] + hessian(x)[1, 1] + 0.5 * ((jacobian(x)[0])**2 + (jacobian(x)[1])**2)) / N
    #print("total", total)
    return total

print(J(np.ones(12)))

steps = 100
theta = jnp.array(np.random.uniform(0, 2, size=12))
learning_rate = 0.2

grad_J = jacfwd(J)

x0 = jnp.array(np.random.uniform(0, 2, size=12))

def id(x):
    return x**2
#res = minimize(J, x0, method='BFGS', options={'disp': True})
#print(res.x)


for i in range(steps):
    print("step ", i)
    a = learning_rate * grad_J(theta)
    if (jnp.any(jnp.isnan(a))):
        print("Error")
        quit()
    else:
        print("THETA", theta)
        theta -= a
print(theta)

