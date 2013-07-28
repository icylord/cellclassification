__author__ = 'shengyinwu'

import numpy as np

""" Aggregating local descriptors into a compact image representation
    Hervé Jégou, Matthijs Douze, Cordelia Schmid and Patrick Pérez
    Proc. IEEE CVPR‘10, June, 2010.
"""

def vl_coding(data, codebook):
    dictionary_size = codebook.shape[0]
    feature_length = codebook.shape[1]

    dot_product = np.dot(data, codebook.T)
    codebook_indexes = np.sort(dot_product, axis = 1)

    vlad_feature = np.zeros((1, feature_length * dictionary_size), np.float)
    codebook_count = np.zeros((1, dictionary_size), np.int)

    for idx in range(0, codebook_indexes.shape[0]):
        codebook_index = codebook_indexes[idx]
        codebook_count[0, codebook_index] = codebook_count[0, codebook_index] + 1
        vlad_feature[0, codebook_index * feature_length:(codebook_index+1) * feature_length] += 
        data[idx, :] - codebook_size[codebook_index, :]
    for codebook_index in range(0, dictionary_size):
        if codebook_count[0, codebook_index] == 0:
            continue
        vlad_feature[0, codebook_index * feature_length:(codebook_index+1) * feature_length] /= 
        codebook_count[0, codebook_index]

    vlad_feature.shape = 1, -1
    vlad_feature = vlad_feature / (np.linalg.norm(vlad_feature) + 1e-12)
    return vlad_feature
