import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import NonUniformImage


N = 100000
M = 101

xs = np.zeros(N)
ys = np.zeros(N)

unis = np.random.uniform(low=0, high=1, size=N)

for i in range(N):
    if (unis[i] > 0.5):
        xs[i] = np.random.normal(5, 1)
        ys[i] = np.random.normal(5, 1)
    else:
        xs[i] = np.random.normal(-5, 1)
        ys[i] = np.random.normal(-5, 1)

H, xedges, yedges = np.histogram2d(xs, ys, bins=(np.linspace(-8, 8, 100), np.linspace(-8, 8, 100)))
# Histogram does not follow Cartesian convention (see Notes),
# therefore transpose H for visualization purposes.
H = H.T

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(title='Heatmap of the initial samples',
        aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
im = NonUniformImage(ax, interpolation='bilinear', )
xcenters = (xedges[:-1] + xedges[1:]) / 2
ycenters = (yedges[:-1] + yedges[1:]) / 2
im.set_data(xcenters, ycenters, H)
ax.images.append(im)
plt.show()
