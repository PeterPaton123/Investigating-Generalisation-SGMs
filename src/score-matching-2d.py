from multiprocessing.spawn import import_main_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from copy import copy
from jax import jacfwd, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp

N = 100000
M = 10

true_mu_1 = np.array([-5., -5.])
cov_1 = np.array([[1., 0.], [0., 1.]])
true_mu_2 = np.array([5., 5.])
cov_2 = np.array([[1., 0.], [0., 1.]])

def f1(v):
    return -(0.5) * (v - true_mu_1) @ cov_1 @ (v - true_mu_1).T

def f2(v):
    return -(0.5) * (v - true_mu_2) @ cov_2 @ (v - true_mu_2).T

def score_fun(v):
    return jnp.log(jnp.exp(f1(v)) + jnp.exp(f2(v)))

def grad_score(v):
    return jacfwd(score_fun)(np.reshape(v, (2,)))

x = np.linspace(-10, 10, 16)
x, y = np.meshgrid(x, x)
grid = np.stack([x.flatten(), y.flatten()], axis=1)
scores = vmap(grad_score)(grid)

fig, axs = plt.subplots(1, 3)

magnitude = [np.sqrt(x**2 + y**2) for [x, y] in scores] 
magnitude = [0 if np.isnan(mag) else mag for mag in magnitude]

norm = colors.Normalize()
norm.autoscale(magnitude)
cm = plt.cm.Reds

sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

fig.suptitle('Languevin dynamics')
for i in range(3):
    axs[i].set_title("T = " + str(i * 0.25))
    axs[i].quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1], color=cm(norm(magnitude)))
    axs[i].scatter([-5, 5], [-5, 5], color='r')
fig.colorbar(sm, label = "Magnitude of the grad modelled score function")

samples = np.resize(np.random.uniform(-7, 7, size=100), (2, 50))

T = 502
dt = 0.003

for t in range(T-1):
    samples += vmap(grad_score)(samples.T).T * dt + np.sqrt(dt * 2) * np.resize(np.random.normal(loc=0, scale=1, size=100), (2, 50))
    if (t % 250 == 0):
        axs[int(t / 250)].scatter(samples[0, :], samples[1, :], color="royalblue", marker=".")

plt.show()

"""
vcolors = ["firebrick", "orange", "olive", "seagreen", "turquoise", "darkcyan", "steelblue", "royalblue", "mediumpurple", "orchid"]

for i in range(5):
    for j in range(10):
        axs[i].scatter(samples[0, j, min(i * 250, 999)], samples[1, j, min(i * 250, 999)], color=vcolors[j])


ics  =      [[2,4],[2,0],[1,4], [2,4],[-2,4],[-2,0],[-1,4],[-2,4], [-2,1],[2,-1]]

for i, ic in enumerate(ics):    
    ax.plot(x[:,0], x[:,1], color=vcolors[i], label='X0=(%.f, %.f)' %  (ic[0], ic[1]) )

ic_x = [ic[0] for ic in ics]
ic_y = [ic[1] for ic in ics]
ax.scatter(ic_x, ic_y, color='blue', s=20)

plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-3.1,3.1)
plt.ylim(-6,6)
plt.legend()
plt.show()

quit()

dev_score = jacfwd(score)

for i in range(M):
    print(i)
    for j in range(M):
        ans = dev_score(np.reshape([xs[i], ys[j]], (2,)))
        us[i, j] = ans[0]
        vs[i, j] = ans[1]

cmap = copy(plt.cm.magma).reversed()
cmap.set_bad(cmap(0))

plt.streamplot(xs, ys, us, vs, cmap=cmap, norm=LogNorm())
#plt.colorbar(pcm, label="J", pad=0)
plt.show()
"""