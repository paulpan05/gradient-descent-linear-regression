from compute_cost import compute_cost
import numpy as np

def gradient_descent(X: np.array, y: np.array, theta: np.array, alpha: float, num_iters: int):
    m = len(y)
    J_history = np.zeros((num_iters, 1))
    for iter in range(num_iters):
        theta = theta - (alpha / m) * X.T.dot((X.dot(theta) - y))
        J_history[iter] = compute_cost(X, y, theta)
    return [theta, J_history]