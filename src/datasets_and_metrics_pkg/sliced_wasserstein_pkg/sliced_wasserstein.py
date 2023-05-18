import jax.numpy as jnp
import numpy as np
from scipy.stats import wasserstein_distance
from jax import random, vmap
import jax

def wasserstein_distance_fun(a, b):
    return wasserstein_distance(u_values=a, v_values=b)

def sliced_wasserstein(test_samples, generated_samples, num_projections, key=random.PRNGKey(0)):
    data_dimension = test_samples.shape[1]
    direction_vectors = random.normal(key, (data_dimension, num_projections))
    direction_vectors /= vmap(jnp.linalg.norm)(direction_vectors.T).T
    projections_test_samples = jnp.dot(test_samples, direction_vectors)
    projections_generated_samples = jnp.dot(generated_samples, direction_vectors)
    result = np.zeros(num_projections)
    for i in range(num_projections):
        result[i] = wasserstein_distance(projections_test_samples[:, i], projections_generated_samples[:, i])
    return jnp.mean(result)