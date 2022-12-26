import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

# Define vector field
def vField(x,t):
    u = x[0]*(x[1]-1)
    v = 4-(x[0])**2-(x[1])**2
    return [u,v]

# Plot vector field
X, Y = np.mgrid[-np.pi:np.pi:-30j,-6:6:30j]
U, V = vField([X,Y],0)

fig, ax = plt.subplots(figsize=(10, 10))
ax.quiver(X, Y, U, V)
plt.show()
quit()

ics  =      [[2,4],[2,0],[1,4], [2,4],[-2,4],[-2,0],[-1,4],[-2,4], [-2,1],[2,-1]]
durations = [[0,4],[0,8],[0,12],[0,4],[0,4],[0,8],[0,12],[0,4],[0,12],[0,12]]
vcolors = plt.cm.autumn_r(np.linspace(0.5, 1., len(ics)))  # colors      for each trajectory

# plot trajectories
for i, ic in enumerate(ics):
    t = np.linspace(durations[i][0], durations[i][1],100)
    x = odeint(vField, ic, t)
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