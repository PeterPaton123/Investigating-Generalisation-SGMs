import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import manifold
from sklearn.datasets import make_swiss_roll
from sklearn.linear_model import LinearRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

num_samples = 20_000
sr_points, sr_color = make_swiss_roll(n_samples=num_samples, random_state=0, hole=False)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
fig.add_axes(ax)
ax.scatter(
    sr_points[:, 0], sr_points[:, 1], sr_points[:, 2], c=sr_color, s=10, alpha=0.7
)
ax.set_title("Swiss Roll in Ambient Space")
ax.view_init(azim=-66, elev=12)
fig.savefig("swiss_roll.png")
quit()
sr_lle, sr_err = manifold.locally_linear_embedding(
    sr_points, n_neighbors=12, n_components=2
)

sr_tsne = manifold.TSNE(n_components=2, perplexity=40, random_state=0).fit_transform(
    sr_points
)

fig, axs = plt.subplots(figsize=(8, 8), nrows=2)
axs[0].scatter(sr_lle[:, 0], sr_lle[:, 1], c=sr_color)
axs[0].set_title("LLE Embedding of Swiss Roll")
axs[1].scatter(sr_tsne[:, 0], sr_tsne[:, 1], c=sr_color)
_ = axs[1].set_title("t-SNE Embedding of Swiss Roll")

plt.show()

embedding = SpectralEmbedding(n_components=2, affinity="rbf")
X_transformed = embedding.fit_transform(sr_points)

sorted_indices = np.argsort(X_transformed[:, 0])

def f(x, model_in):
    coefficients = model_in.named_steps['linear'].coef_
    res = 0.0
    for i in range(len(coefficients)):
        res += x**i * coefficients[i]
    return res
    #return model.named_steps['linear'].coef_[2] * x**2 + model.named_steps['linear'].coef_[1] * x + model.named_steps['linear'].coef_[0]

X_transformed = X_transformed[sorted_indices, :]
X_transformed[:, 0] = (X_transformed[:, 0] - np.min(X_transformed[:, 0])) / (np.max(X_transformed[:, 0]) - np.min(X_transformed[:, 0]))
X_transformed[:, 1] = (X_transformed[:, 1] - np.min(X_transformed[:, 1])) / (np.max(X_transformed[:, 1]) - np.min(X_transformed[:, 1]))

model = Pipeline([('poly', PolynomialFeatures(degree=4)),
                  ('linear', LinearRegression(fit_intercept=False))])

model = model.fit(X_transformed[:, 0, np.newaxis], X_transformed[:, 1])
print(model.named_steps['linear'].coef_)

dummy_in = np.zeros((1000, 2))
dummy_in[:, 0] = np.linspace(X_transformed[0, 0], X_transformed[2000, 0], 1000)
dummy_in[:, 1] = -np.linspace(X_transformed[0, 0] + 0.002, X_transformed[2000, 0] + 0.002, 1000)
plt.title("Embedding")
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=sr_color[sorted_indices], s=30, alpha=0.8)
plt.plot(np.linspace(X_transformed[0, 0], X_transformed[-1, 0], 1000), f(np.linspace(X_transformed[0, 0], X_transformed[-1, 0], 1000), model), linestyle='--', c='r')
plt.plot(dummy_in[:, 0], dummy_in[:, 1])
plt.show()

def swiss_roll_generalisation_metric(transformed_in):
    return 5.0 * wasserstein_distance(transformed_in[:, 0], X_transformed[:, 0]), np.mean(np.abs(model.predict(transformed_in[:, 0, np.newaxis]), transformed_in[:, 1]))
