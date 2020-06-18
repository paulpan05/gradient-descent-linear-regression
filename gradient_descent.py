import numpy as np
from plot_data import plot_data
from compute_cost import compute_cost

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

m = len(y)

plot_data(X, y)

X = np.c_[np.ones((m, 1)), data[:, 0]]
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

J = compute_cost(X, y, theta)

J1 = compute_cost(X, y, np.mat('-1 ; 2'))