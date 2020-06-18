import numpy as np
import matplotlib.pyplot as plt

def plot_data(x: np.array, y: np.array, delimiter: str):
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    if delimiter == 'x':
        plt.scatter(x, y, s=10, marker='x', c='red')
    else:
        plt.plot(x, y)
    plt.draw()