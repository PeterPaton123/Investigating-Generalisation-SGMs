from sklearn.datasets import make_swiss_roll
import sys
sys.path.append("/home/peter/Documents/Year-4/fyp/Numerical-methods-for-score-based-modelling/src/")
from datasets_and_metrics_pkg import make_severed_sphere, sliced_wasserstein
import matplotlib.pyplot as plt
import jax.random as random
import jax.numpy as jnp

true_swiss_roll, true_swiss_roll_colours = make_swiss_roll(n_samples=5000, random_state=0)
true_swiss_roll_2, true_swiss_roll_colours_2 = make_swiss_roll(n_samples=5000, random_state=10)
gapped_swiss_roll, gapped_swiss_roll_colours = make_swiss_roll(n_samples=5000, random_state=2, hole=True)
noisy_swiss_roll, noisy_swiss_roll_colours = make_swiss_roll(n_samples=5000, noise=1.5, random_state=1)
severed_sphere, severed_sphere_colours = make_severed_sphere(5000, random.PRNGKey(0), p_reduce=0.0, t_reduce=0.0)
"""
fig = plt.figure()
ax_0 = fig.add_subplot(1, 1, 1, projection='3d')
ax_0.scatter(true_swiss_roll[:, 0], true_swiss_roll[:, 1], true_swiss_roll[:, 2], c=true_swiss_roll_colours, s=10, alpha=0.8)
ax_0.set_title("Swiss Roll")
ax_0.view_init(azim=-66, elev=12)
fig.savefig("swiss_roll.png")

fig = plt.figure()
ax_1 = fig.add_subplot(1, 1, 1, projection='3d')
ax_1.scatter(gapped_swiss_roll[:, 0], gapped_swiss_roll[:, 1], gapped_swiss_roll[:, 2], c=gapped_swiss_roll_colours, s=10, alpha=0.8)
ax_1.set_title("Gapped Swiss Roll")
ax_1.view_init(azim=-66, elev=12)
fig.savefig("gapped_swiss_roll.png")

fig = plt.figure()
ax_2 = fig.add_subplot(1, 1, 1, projection='3d')
ax_2.scatter(noisy_swiss_roll[:, 0], noisy_swiss_roll[:, 1], noisy_swiss_roll[:, 2], c=noisy_swiss_roll_colours, s=10, alpha=0.8)
ax_2.set_title("Noisy Swiss Roll")
ax_2.view_init(azim=-66, elev=12)
fig.savefig("noised_swiss_roll.png")

fig = plt.figure()
ax_3 = fig.add_subplot(1, 1, 1, projection='3d')
ax_3.scatter(severed_sphere[:, 0], severed_sphere[:, 1], severed_sphere[:, 2], c=severed_sphere_colours, s=10, alpha=0.8)
ax_3.set_title("Sphere")
ax_3.view_init(azim=-66, elev=12)
fig.savefig("sphere.png")
"""
print(f"swiss roll and another swiss roll: {sliced_wasserstein(jnp.array(true_swiss_roll), jnp.array(true_swiss_roll_2), 100)}")
print(f"swiss roll and gapped swiss roll: {sliced_wasserstein(jnp.array(true_swiss_roll), jnp.array(gapped_swiss_roll), 100)}")
print(f"swiss roll and noisy swiss roll: {sliced_wasserstein(jnp.array(true_swiss_roll), jnp.array(noisy_swiss_roll), 100)}")
print(f"swiss roll and severed sphere: {sliced_wasserstein(jnp.array(true_swiss_roll), severed_sphere, 100)}")