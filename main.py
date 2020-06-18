import numpy as np
from matplotlib import pyplot as plt
from plot_data import plot_data
from compute_cost import compute_cost
from gradient_descent import gradient_descent

data = np.loadtxt('ex1data1.txt', delimiter=',')

X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

m = len(y)

plot_data(X, y, 'x')

X = np.c_[np.ones((m, 1)), data[:, 0]]
theta = np.zeros((2, 1))

iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

J = compute_cost(X, y, theta)

print('With theta = [0 ; 0]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 32.07\n')

J = compute_cost(X, y, np.mat('-1 ; 2'))

print('With theta = [-1 ; 2]\nCost computed = %f\n' % J)
print('Expected cost value (approx) 54.24\n')

[theta, J_history] = gradient_descent(X, y, theta, alpha, iterations)

print('Theta found by gradient descent:\n')
print('%s\n' % theta)
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

plot_data(X[:, 1], X.dot(theta), '-')

plt.show()

predict1 = np.array([1, 3.5]).dot(theta)

print('For population = 35,000, we predict a profit of %f\n' % np.sum(predict1*10000))

predict2 = np.array([1, 7]).dot(theta)

print('For population = 70,000, we predict a profit of %f\n' % np.sum(predict2*10000))