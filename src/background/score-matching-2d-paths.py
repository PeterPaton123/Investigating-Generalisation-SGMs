from multiprocessing.spawn import import_main_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from jax import jacfwd, vmap
import jax.numpy as jnp

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

magnitude = [np.sqrt(x**2 + y**2) for [x, y] in scores] 
magnitude = [0 if np.isnan(mag) else mag for mag in magnitude]

norm = colors.Normalize()
norm.autoscale(magnitude)
cm = plt.cm.Reds

sm = plt.cm.ScalarMappable(cmap=cm, norm=norm)

plt.quiver(grid[:, 0], grid[:, 1], scores[:, 0], scores[:, 1], color=cm(norm(magnitude)))
plt.scatter([-5, 5], [-5, 5], color='r')
samples = np.ones((2, 10, 1002))
samples[:, :, 0] = np.resize(np.random.uniform(-7, 7, size=20), (2, 10))


T = 1002
dt = 0.003

for t in range(T-1):
    samples[:, :, t+1] = samples[:, :, t] + vmap(grad_score)(samples[:, :, t].T).T * dt + np.sqrt(dt * 2) * np.resize(np.random.normal(loc=0, scale=1, size=20), (2, 10))

vcolors = ["firebrick", "orange", "olive", "seagreen", "turquoise", "darkcyan", "steelblue", "royalblue", "mediumpurple", "orchid"]

for i in range(3):
    plt.scatter(samples[0, i, 0], samples[1, i, 0], color='k')
    x_smooth = samples[0, i, ::10]
    y_smooth = samples[1, i, ::10]
    plt.plot(x_smooth, y_smooth, color='k', alpha=0.5)

plt.legend(loc="lower left")
plt.show()

"""
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