# This plotting code is largely based on code from the sklearn documentation
# https://scikit-learn.org/stable/auto_examples/manifold/plot_manifold_sphere.html

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from sklearn import manifold

def plot_severed_sphere(samples, fname="", num_neighbours=25, colours=jnp.empty(shape=(0))):
    num_samples = samples.shape[0]
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    if colours.size == 0:
        colours = z
    
    fig = plt.figure(figsize=plt.figaspect(1.0/3))
    plt.suptitle("Manifold Learning with %i points, %i neighbors" % (num_samples, num_neighbours), fontsize=14)

    ax_0 = fig.add_subplot(1, 3, 1, projection='3d')
    ax_0.scatter(x, y, z, c=colours, cmap=plt.cm.rainbow, alpha=0.5)
    ax_0.view_init(40, -10)

    sphere_data = np.array([x, y, z]).T
    ax_1 = fig.add_subplot(1, 3, 2)
    trans_data = (
        manifold.LocallyLinearEmbedding(
            n_neighbors=num_neighbours, n_components=2, method="modified"
        )
        .fit_transform(sphere_data)
        .T)
    ax_1.scatter(trans_data[0], trans_data[1], c=colours, cmap=plt.cm.rainbow)
    ax_1.set_title("Modified LLE")

    ax_2 = fig.add_subplot(1, 3, 3)
    se = manifold.SpectralEmbedding(n_components=2, n_neighbors=num_neighbours)
    trans_data = se.fit_transform(sphere_data).T
    ax_2.scatter(trans_data[0], trans_data[1], c=colours, cmap=plt.cm.rainbow)
    ax_2.set_title("Spectral Embedding")
    fig.savefig(f"{fname}.png")