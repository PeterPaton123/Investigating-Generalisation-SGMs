import jax.numpy as jnp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pdf_utils import pdf_normal
from jax import vmap

class InteractingSDE():

    """
    Numerically solves mutlivariate stochastic differential equations of the form:

    dXn(t) = (A1 * Xn(t) + A2) dt + B dW(t)
    Xn(t) is a column vector in R^n
    A1 is an n x n matrix
    At is a function which takes in t and x and returns a n x 1 column vector
    B is an n x m matrix
    dW(t) ~ N(0m, sqrt(dt) * Idm)
    """

    def __init__(self, x0s, dt, A1, A2, B):
        (self.n, self.m) = jnp.shape(B)
        self.samples = jnp.resize(jnp.array(x0s), (self.n, 1))
        self.dt = dt
        self.A1 = jnp.resize(A1, (self.n, self.n))
        self.A2 = A2
        self.B = jnp.resize(B, (self.n, self.m))
        self.T = 0
        self.ts = jnp.array([self.T])

    """
    Perform a step using Euler Maruyama discretisation
    """
    def step_euler_maruyama(self):
        #print(self.samples[:, -1])
        prevXs = jnp.resize(self.samples[:, -1], (self.n, 1))
        new_time = float(self.ts[-1] + self.dt)
        brownian_motion_samples = jnp.resize(np.random.normal(loc=0, scale=jnp.sqrt(self.dt), size=self.m), (self.m, 1))
        #print(jnp.matmul(self.A1, prevXs))
        drift = jnp.matmul(self.A1, prevXs)
        noise = jnp.matmul(self.B, brownian_motion_samples)
        term = jnp.resize(self.A2(prevXs, new_time), (self.n, 1))
        samples_at_t_plus_dt = prevXs + (drift + term) * self.dt + noise
        self.samples = jnp.append(self.samples, samples_at_t_plus_dt, axis=1)
        print("SHAPE: prevXs", jnp.shape(prevXs), "drift ", jnp.shape(drift), "noise: ", jnp.shape(noise), "term: ", jnp.shape(term), "new samples", jnp.shape(samples_at_t_plus_dt), "samples", jnp.shape(self.samples))
        self.ts = jnp.append(self.ts, self.ts[-1] + self.dt)
        self.T += self.dt

    def step_up_to_T(self, T : float):
        t0 = self.T
        pdf = vmap(lambda x: 0.5 * pdf_normal(-5, 1, x) + 0.5 * pdf_normal(5, 1, x))(np.linspace(-10, 10, 100))
        for i in range(0, int((T - t0) / self.dt)):
            if (i % 10 == 0):
                #print(self.samples[:, -1])
                
                plt.clf()
                plt.plot(np.linspace(-10, 10, 100), pdf, linestyle='-', color='b')
                #plt.hist(self.samples[:, -1], bins=40, density=True, stacked=True)
                plt.plot(self.samples[:, -1], np.zeros(self.n), marker='o', color='r')
                #plt.scatter(self.samples[:, -2], np.zeros(self.n), s=5, color='r')
                plt.show(block=False)
                plt.pause(0.01)
                
            self.step_euler_maruyama()
        plt.hist(jnp.resize(self.samples[:, -1], (1, self.n)), bins=20, density=True)
        plt.plot(np.linspace(-3, 3, 30), vmap(lambda x: pdf_normal(0,1, x))(np.linspace(-3, 3, 30)), linestyle='-', color='r')
        plt.show()
        