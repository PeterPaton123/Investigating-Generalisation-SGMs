import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import manifold
from sklearn.datasets import make_swiss_roll
from sklearn.linear_model import LinearRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def kernel(x, y):
    return np.exp(-0.5 * np.linalg.norm(x - y)**2)

def MMD(generated_samples, test_samples):
    """
    Maximum mean discrepancy
    """
    N = np.shape(generated_samples)[0]
    M = np.shape(test_samples)[0]
    return (1.0 / N)**2 * np.sum([[kernel(generated_samples[i, :], generated_samples[j, :]) for j in range(N)] for i in range(N)]) + \
        (1.0 / M)**2 * np.sum([[kernel(test_samples[i, :], test_samples[j, :]) for j in range(M)] for i in range(M)]) + \
        (1.0 / (N * M)) * np.sum([[kernel(generated_samples[i, :], test_samples[j, :]) for j in range(M)] for i in range(N)])

class ManifoldMetric():

    def __init__(self, input_points, input_color, model_degree=4):
        # Data embedding
        self.embedding = SpectralEmbedding(n_components=2, affinity="rbf")
        input_transformed = self.embedding.fit_transform(input_points)
        
        # Embedding pre-processing
        sorted_indices = np.argsort(input_transformed[:, 0])
        input_transformed = input_transformed[sorted_indices, :]
        self.min_x = np.min(input_transformed[:, 0])
        self.max_x = np.max(input_transformed[:, 0])
        self.min_y = np.min(input_transformed[:, 1])
        self.max_y = np.max(input_transformed[:, 1])
        self.input_transformed = self.normalise(input_transformed)

        # Fit model to embedding
        self.model = Pipeline([('poly', PolynomialFeatures(degree=model_degree)),
                        ('linear', LinearRegression(fit_intercept=False))])
        self.model = self.model.fit(self.input_transformed[:, 0, np.newaxis], self.input_transformed[:, 1])

    def normalise(self, in_data):
        in_data[:, 0] = (in_data[:, 0] - self.min_x) / (self.max_x - self.min_x)
        in_data[:, 1] = (in_data[:, 1] - self.min_y) / (self.max_y - self.min_y)
        return in_data

    def generalisation_metric(self, generated_samples):
        # Find embedding of generated samples
        generated_embedding = SpectralEmbedding(n_components=2, affinity="rbf")
        generated_transformed = generated_embedding.fit_transform(generated_samples)
        print("Embedding found")
        # Pre-process the embedding
        generated_transformed = self.normalise(generated_transformed)
        print("Embedding pre-processed, returning result")
        # Calculate generalisation metric on embedding
        return 10.0 * wasserstein_distance(self.input_transformed[:, 0], generated_transformed[:, 0]), np.mean(np.abs(self.model.predict(generated_transformed[:, 0, np.newaxis]), generated_transformed[:, 1]))
    