import numpy as np
from scipy.optimize import minimize
import pylab as pl

n_samples = 20000
np.random.seed(0)
X = np.zeros((n_samples*2, 3), np.float)
X[0:n_samples, :] = np.random.randn(n_samples, 3) * [1, 1, 0] + [-2, 2, 1]
X[n_samples:n_samples*2, :] = np.random.randn(n_samples, 3) * [1, 1, 0] + [2, -2, 1]
y = np.array([-1] * n_samples + [1] * n_samples)
C = 0.1

def l2loss_svm(W):
    yp = np.dot(X, W.T)
    idx = np.nonzero(yp * y < 1)[0]
    e = yp[idx] - y[idx]
    f = np.dot(e.T, e) + C * np.dot(W.T, W)
    return f

def l2loss_svm_der(W):
    yp = np.dot(X, W.T)
    idx = np.nonzero(yp * y < 1)[0]
    e = yp[idx] - y[idx]
    df = np.array([0, 0, 0])
    df[0:2] = 2 * np.dot(X[idx, 0:2].T, e) + 2 * C * W[0:2]
    df[2] = 2 * np.dot(X[idx, 2].T, e)
    return df


def main():
    W = np.array([0, 0, 0], np.float)
    xopt = minimize(l2loss_svm, W, jac=l2loss_svm_der)

    # get the separating hyperplane
    w = xopt.x
    a = -w[0] / w[1]
    xx = np.linspace(-5, 5)
    yy = a * xx - w[2] / w[1]

    # plot the line, the points, and the nearest vectors to the plane
    pl.plot(xx, yy, 'k-')
    pl.scatter(X[:, 0], X[:, 1], c=y, cmap=pl.cm.Paired)

    pl.axis('tight')
    pl.show()

if __name__ == '__main__':
    main()