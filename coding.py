__author__ = 'shengyinwu'

import numpy as np

"""
Aggregating local descriptors into a compact image representation
Hervé Jégou, Matthijs Douze, Cordelia Schmid and Patrick Pérez
Proc. IEEE CVPR‘10, June, 2010.
"""

def vlad_coding(data, codebook):
    dictionary_size = codebook.shape[0]
    feature_length = codebook.shape[1]

    codebook_indexes = np.argmax(np.dot(data, codebook.T), axis = 1)
    vlad_feature = np.zeros((1, feature_length * dictionary_size), np.float)

    for codebook_index in range(0, dictionary_size):
        current_indexs = np.nonzero(codebook_indexes == codebook_index)[0]
        if current_indexs.shape[0] == 0:
            continue
        tile_centroid = np.tile(codebook[codebook_index, :], (current_indexs.shape[0], 1))
        vlad_feature[0, codebook_index * feature_length:(codebook_index+1) * feature_length] = \
            (data[current_indexs, :] - tile_centroid).sum(0) / current_indexs.shape[0]

    vlad_feature.shape = 1, -1
    return vlad_feature / (np.linalg.norm(vlad_feature) + 1e-12)
