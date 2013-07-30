__author__ = 'shengyinwu'

import numpy as np

def vl_coding(data, codebook):
    dictionary_size = codebook.shape[0]
    feature_length = codebook.shape[1]

    dot_product = np.dot(data, codebook.T)
    codebook_indexes = np.argmax(dot_product)
    vlad_feature = np.zeros((1, feature_length * dictionary_size), np.float)

    for codebook_index in range(0, dictionary_size):
        current_indexs = np.nonzero(codebook_indexes == codebook_index)[0]
        if current_indexs.shape[0] == 0:
            continue
        current_centroid = codebook[codebook_index, :]
        tile_centroid = np.tile(current_centroid, (current_indexs.shape[0], 1))
        vlad_feature[0, codebook_index * feature_length:(codebook_index+1) * feature_length] = \
            (data[current_indexs, :] - tile_centroid).sum(0) / current_indexs.shape[0]

    vlad_feature.shape = 1, -1
    vlad_feature = vlad_feature / (np.linalg.norm(vlad_feature) + 1e-12)
    return vlad_feature
