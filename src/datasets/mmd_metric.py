import numpy as np

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
