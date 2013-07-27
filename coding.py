__author__ = 'shengyinwu'

import numpy as np

def vlad_coding(X, B, Width, Height, Xs, Ys):
    dSize = B.shape[0]
    nSmp = X.shape[0]
    feature_len = B.shape[1]

    img_width = Width
    img_height = Height

    D = np.dot(X, B.T)
    IDX = np.sort(D, axis = 1)

    beta = np.zeros((1, feature_len * dSize), np.float)
    count = np.zeros((1, dSize), np.int)

    for idx in range(0, IDX.shape[0]):
        index = IDX[idx, 0]
        count[0, index] = count[0, index] + 1
        beta[0, index * feature_len:(index+1) * feature_len] += X[idx, :] - B[index, :]
    for index in range(0, dSize):
        if count[0, index] == 0:
            continue
        beta[0, index * feature_len:(index+1) * feature_len] /= count[0, index]

    beta.shape = 1, -1
    beta = beta / (np.linalg.norm(beta) + 1e-12)
    return beta[0, :]
