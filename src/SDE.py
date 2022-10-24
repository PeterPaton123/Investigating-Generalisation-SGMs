from abc import abstractmethod
from jax import jit
import jax.numpy as jnp
from prior import mixture_prior
import numpy as np
from jax.tree_util import Partial
from plotting import plot

import typing

"""
Class defining a stochastic differential equation of the form:
dX(t) = u(t, X(t)) * dt + s(t, X(t)) * dWt
"""
class SDE():

    """
    u : a function of t and xt
    s : a function of t and xt
    expected_pdf: Expected pdf P(X(t) = xt) at time t (a function of t and xt)
    """
    def __init__(self, x0s, dt, u, s):
        self.samples = np.resize(np.array(x0s), (np.size(x0s), 1))
        self.ts = np.array([0])
        self.dt = dt
        self.T = 0
        self.u = u
        self.s = s

    """
    Perform a step using a Euler Maruyama discretisation
    """
    def step_euler_maruyama(self):
        prevXs = self.samples[:, -1]
        t = self.T + self.dt
        uPartial = Partial(self.u, t)
        sPartial = Partial(self.s, t)
        brownian_motion_samples = np.random.normal(loc = 0, scale = np.sqrt(self.dt), size=np.size(prevXs))
        samples_at_t_plus_dt = prevXs + self.dt * uPartial(prevXs) + np.multiply(brownian_motion_samples, sPartial(prevXs))
        self.samples = np.column_stack((self.samples, samples_at_t_plus_dt))
        self.ts = np.append(self.ts, t)
        self.T += self.dt

    def step_up_to_T(self, T : float):
        t0 = self.T
        for i in range(0, (int) ((T - t0) / self.dt)):
            self.step_euler_maruyama()
