__author__ = 'ShengyinWu'

import numpy as np
import os
import Image

from coding import *

class BOW:
    def __init__(self, descriptor_dir, bow_dir, codebook):
        self.descriptor_dir = descriptor_dir
        self.bow_dir = bow_dir
        self.codebook = codebook

    def doBOW(self):
        if not os.path.exists(self.bow_dir):
            os.makedirs(self.bow_dir)
        files_in_dir = os.listdir(self.descriptor_dir)
        codebook_file = os.path.join("data", "words", "[{0}].npy".format(self.codebook))
        fd = file(codebook_file, "rb")
        codebook = np.load(fd)
        fd.close()
        for f in files_in_dir:
            print f
            if f[-1] == "y":
                npy_file = os.path.join(self.descriptor_dir, f)
                # may be some files broken
                if os.path.getsize(npy_file) < 300:
                    continue
                fd = file(npy_file)
                descriptors = np.load(fd)

                data_norm = np.sum(np.abs(descriptors)**2, axis=-1)**(1./2) + 1e-12
                tile_data_norm = np.tile(data_norm, (descriptors.shape[1], 1))
                descriptors = descriptors / tile_data_norm.T
                fd_feature = file(os.path.join(self.bow_dir, f), "wb")
                fea = vlad_coding(descriptors, codebook)
                np.save(fd_feature, fea)

def main():
    word = 256
    train_bow = BOW(
        os.path.join("data", "desc", "training_desc"),
        os.path.join("data", "bow", "vlad_train{0}".format(word)), word)
    train_bow.doBOW()
    test_bow = BOW(
        os.path.join("data", "desc", "test_desc"),
        os.path.join("data", "bow", "vlad_test{0}".format(word)), word)
    test_bow.doBOW()

if __name__ == "__main__":
    main()