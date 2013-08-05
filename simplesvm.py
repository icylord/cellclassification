import numpy as np
from scipy.optimize import fmin_bfgs
import pylab as pl

n_samples = 20
np.random.seed(0)
X = np.zeros((n_samples*2, 3), np.float)
X[0:n_samples, :] = np.random.randn(n_samples, 3) * [1, 1, 0] + [-2, -2, 1]
X[n_samples:n_samples*2, :] = np.random.randn(n_samples, 3) * [1, 1, 0] + [2, 2, 1]
y = np.array([-1] * n_samples + [1] * n_samples)
C = 1

def LSSVM(W):
    yp = np.dot(X, W.T)
    idx = np.nonzero(yp * y < 1.0)[0]
    e = yp[idx] - y[idx]
    f = np.dot(e.T, e) + C * np.dot(W.T, W)
    return f

def LSSVM_DER(W):
    yp = np.dot(X, W.T)
    idx = np.nonzero(yp * y < 1.0)[0]
    e = yp[idx] - y[idx]
    df = np.array([0, 0, 0])
    df[0:2] = 2 * (np.dot(X[idx, 0:2].T, e) + C * W[0:2])
    df[2] = 2 * np.dot(X[idx, 2].T, e)
    return df


def main():
    W = np.array([0, 0, 0], np.float)
    xopt = fmin_bfgs(LSSVM, W, fprime=LSSVM_DER)
    # get the separating hyperplane
    w = xopt
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - w[2] / w[1]
    print w

    # plot the line, the points, and the nearest vectors to the plane
    pl.plot(xx, yy, 'k-')

    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired)

    pl.axis('tight')
    pl.show()

if __name__ == '__main__':
    main()