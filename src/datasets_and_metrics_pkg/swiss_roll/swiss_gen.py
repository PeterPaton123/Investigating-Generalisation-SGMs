from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

sr_samples, sr_color = make_swiss_roll(n_samples=7000, random_state=0)
fig.add_subplot(1, 2, 1, projection='3d')
ax_0.scatter(sr_samples[:, 0], sr_samples[:, 1], sr_samples[:, 2], c=sr_color, s=10, alpha=0.8)
ax_0.set_title("Generated Swiss Roll")
ax_0.view_init(azim=-66, elev=12)