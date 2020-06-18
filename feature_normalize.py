import numpy as np

def feature_normalize(X):
    X_norm = X
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(np.size(X, 1)):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]
    return [X_norm, mu, sigma]