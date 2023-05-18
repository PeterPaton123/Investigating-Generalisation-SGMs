import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance
from sklearn import manifold
from sklearn.datasets import make_swiss_roll
from sklearn.linear_model import LinearRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

class SwissRollMetric():

    def __init__(self, input_points, model_degree=4):
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
        self.model = Pipeline([
            ('poly', PolynomialFeatures(degree=model_degree)),
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
        # Pre-process the embedding
        generated_transformed = self.normalise(generated_transformed)
        #print("Embedding pre-processed, returning result")
        # Calculate generalisation metric on embedding
        return 10.0 * wasserstein_distance(self.input_transformed[:, 0], generated_transformed[:, 0]), 10.0 * wasserstein_distance(self.input_transformed[:, 1], generated_transformed[:, 1]), np.mean(np.abs(self.model.predict(generated_transformed[:, 0, np.newaxis]), generated_transformed[:, 1]))

