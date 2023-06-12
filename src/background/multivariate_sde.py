
import numpy as np
from jax import jit, vmap
from jax.tree_util import Partial
import matplotlib.pyplot as plt
from wasserstein_distance import ws_dist_normal

class MV_SDE():

    """
    u : a function of t and xt
    s : a function of t and xt
    """
    def __init__(self, x0s, dt, u, s):
        self.samples = x0s[:, :, np.newaxis]
        self.ts = np.array([0])
        self.dt = dt
        self.T = 0
        self.u = u
        self.s = s

    """
    Perform a step using the Euler Maruyama method
    """
    def step_euler_maruyama(self):
        prevXs = self.samples[:, :, -1]
        t = self.T + self.dt
        uPartial = Partial(self.u, t)
        sPartial = Partial(self.s, t)
        brownian_motion_samples = np.random.normal(loc = 0, scale = np.sqrt(self.dt), size=np.shape(prevXs))
        samples_at_t_plus_dt = prevXs + self.dt * uPartial(prevXs) + np.multiply(brownian_motion_samples, sPartial(prevXs))
        self.samples = np.concatenate([self.samples, samples_at_t_plus_dt[..., None]], axis=2)
        self.ts = np.append(self.ts, t)
        self.T += self.dt

    def step_up_to_T(self, T):
        t0 = self.T
        steps = int((T - t0) / self.dt)
        for _t in range(0, steps):
            self.step_euler_maruyama()