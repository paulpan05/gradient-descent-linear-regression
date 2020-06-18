import numpy as np

def compute_cost(X: np.array, y: np.array, theta: np.array):
    m = len(y)
    return np.sum(np.square(X.dot(theta) - y)) / (2 * m)