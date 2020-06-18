import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    return np.sum(np.square(X.dot(theta) - y)) / (2 * m)