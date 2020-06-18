import matplotlib.pyplot as plt

def plot_data(x: [int], y: [int]):
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')
    plt.scatter(x, y, s=10, marker='x', c='red')
    plt.show()