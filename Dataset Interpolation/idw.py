import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import pandas as pd

"""
read data from rssi.csv file and filter 
it to access point A in floor 1
"""
data = pd.read_csv('rssi.csv')
flt = data['ap'] == 'A'
data2 = data[flt]
data3 = data2['z'] == 1.0
data3 = data2[data3]


def main():
    # Setup: Generate data...
    n = 10
    nx, ny = 50, 50
    x = data3['x']
    y = data3['y']
    z = data3['z']
    xi = np.linspace(x.min(), x.max(), nx)
    yi = np.linspace(y.min(), y.max(), ny)
    xi, yi = np.meshgrid(xi, yi)
    xi, yi = xi.flatten(), yi.flatten()

    # Calculate IDW
    grid1 = simple_idw(x, y, z, xi, yi)
    grid1 = grid1.reshape((ny, nx))

    # Comparisons...
    plot(x, y, z, grid1)
    plt.title('wifi fingerprint radio map for AP "A" in floor 1')

    plt.show()


def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


def plot(x, y, z, grid):
    plt.figure()
    plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()))
    # plt.hold(True)
    plt.scatter(x, y, c=z)
    plt.colorbar()


if __name__ == '__main__':
    main()
