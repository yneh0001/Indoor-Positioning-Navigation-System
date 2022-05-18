import numpy as np
from scipy.spatial import cKDTree
import pandas as pd
import openturns as ot

data = pd.read_csv('rssi.csv')
flt = data['ap'] == 'A'
data2 = data[flt]
data3 = data2['z'] == 1.0
data3 = data2[data3]

x = data3['x']
y = data3['y']
z = data3['signal']

list_x = x.tolist()
list_y = y.tolist()
list_z = z.tolist()

x_y = [list(x) for x in zip(list_x, list_y)]


def ftr(xy):
    lst = []
    count = 0
    average = 0
    for i in range(0, len(x) - 1):
        average = list_z[i]
        count = 1
        if x_y[i] == x_y[i + 1]:
            count += 1
            average += list_z[i + 1]
            i += 1
            continue
        else:
            lst.append([x_y[i - 1], average / count])
            count = 0
            average = 0
            i += 1
            continue

    return lst


x_y = ftr(x_y)

new_xy = []
new_z = []


def adjust(l):
    for i in range(0, len(x_y) - 1):
        new_xy.append(x_y[i][0])
        new_z.append(x_y[i][1])
        continue

    return new_xy


xy = adjust(x_y)


def extractDigits(lst):
    return [[el] for el in lst]


new_z = extractDigits(new_z)

new_xy = ot.Sample(xy)
new_z = ot.Sample(new_z)


class tree(object):
    """
    Compute the score of query points based on the scores of their k-nearest neighbours,
    weighted by the inverse of their distances.
    @reference:
    https://en.wikipedia.org/wiki/Inverse_distance_weighting
    Arguments:
    ----------
        X: (N, d) ndarray
            Coordinates of N sample points in a d-dimensional space.
        z: (N,) ndarray
            Corresponding scores.
        leafsize: int (default 10)
            Leafsize of KD-tree data structure;
            should be less than 20.
    Returns:
    --------
        tree instance: object
    Example:
    --------
    # 'train'
    idw_tree = tree(X1, z1)
    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)
    See also:
    ---------
    demo()
    """
    def __init__(self, X=None, z=None, leafsize=10):
        if not X is None:
            self.tree = cKDTree(X, leafsize=leafsize )
        if not z is None:
            self.z = np.array(z)

    def fit(self, X=None, z=None, leafsize=10):
        """
        Instantiate KDtree for fast query of k-nearest neighbour distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N sample points in a d-dimensional space.
            z: (N,) ndarray
                Corresponding scores.
            leafsize: int (default 10)
                Leafsize of KD-tree data structure;
                should be less than 20.
        Returns:
        --------
            idw_tree instance: object
        Notes:
        -------
        Wrapper around __init__().
        """
        return self.__init__(X, z, leafsize)

    def __call__(self, X, k=6, eps=1e-6, p=2, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        """
        self.distances, self.idx = self.tree.query(X, k, eps=eps, p=p)
        self.distances += regularize_by
        weights = self.z[self.idx.ravel()].reshape(self.idx.shape)
        mw = np.sum(weights/self.distances, axis=1) / np.sum(1./self.distances, axis=1)
        return mw

    def transform(self, X, k=6, p=2, eps=1e-6, regularize_by=1e-9):
        """
        Compute the score of query points based on the scores of their k-nearest neighbours,
        weighted by the inverse of their distances.
        Arguments:
        ----------
            X: (N, d) ndarray
                Coordinates of N query points in a d-dimensional space.
            k: int (default 6)
                Number of nearest neighbours to use.
            p: int or inf
                Which Minkowski p-norm to use.
                1 is the sum-of-absolute-values "Manhattan" distance
                2 is the usual Euclidean distance
                infinity is the maximum-coordinate-difference distance
            eps: float (default 1e-6)
                Return approximate nearest neighbors; the k-th returned value
                is guaranteed to be no further than (1+eps) times the
                distance to the real k-th nearest neighbor.
            regularise_by: float (default 1e-9)
                Regularise distances to prevent division by zero
                for sample points with the same location as query points.
        Returns:
        --------
            z: (N,) ndarray
                Corresponding scores.
        Notes:
        ------
        Wrapper around __call__().
        """
        return self.__call__(X, k, eps, p, regularize_by)

def main():
    import matplotlib.pyplot as plt

    def func(x, y):
        return np.sin(x ** 2 + y ** 2) / (x ** 2 + y ** 2)
    # 'train'
    idw_tree = tree(new_xy, new_z)

    # 'test'
    spacing = np.linspace(-5., 5., 100)
    X2 = np.meshgrid(spacing, spacing)
    grid_shape = X2[0].shape
    X2 = np.reshape(X2, (2, -1)).T
    z2 = idw_tree(X2)

    # plot
    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,3))
    ax1.contourf(spacing, spacing, func(*np.meshgrid(spacing, spacing)))
    ax1.set_title('Ground truth')
    ax2.scatter(new_xy[:,0], new_xy[:,1], c=new_z, linewidths=0)
    ax2.set_title('Samples')
    ax3.contourf(spacing, spacing, z2.reshape(grid_shape))
    ax3.set_title('Reconstruction')
    plt.show()

if __name__ == '__main__':
    main()
