__author__ = 'ShengyinWu'

import numpy as np
import os
import Image

from coding import *

class MyBagOfWords:
    def __init__(self, descriptor_dir, bow_dir, num_centers):
        self.descriptor_dir = descriptor_dir
        self.bow_dir = bow_dir
        self.num_centers = num_centers

    def do_bagofwords(self):
        if not os.path.exists(self.bow_dir):
            os.makedirs(self.bow_dir)
        fd = file(os.path.join("data", "words", "{0}.npy".format(self.num_centers)), "rb")
        codebook = np.load(fd)
        fd.close()
        for f in os.listdir(self.descriptor_dir):
            print f
            if f[-1] == "y":
                npy_file = os.path.join(self.descriptor_dir, f)
                # may be some files broken
                if os.path.getsize(npy_file) < 300:
                    continue
                fd = file(npy_file)
                descriptors = np.load(fd)

                tile_data_norm = np.tile(np.sum(np.abs(descriptors)**2, axis=-1)**(1./2) + 1e-12,
                                         (descriptors.shape[1], 1))
                fd_feature = file(os.path.join(self.bow_dir, f), "wb")
                np.save(fd_feature, vlad_coding(descriptors / tile_data_norm.T, codebook))

def main():
    num_centers = 256
    train_bow = MyBagOfWords(
        os.path.join("data", "desc", "training_desc"),
        os.path.join("data", "bow", "vlad_train{0}".format(num_centers)), num_centers)
    train_bow.do_bagofwords()
    test_bow = MyBagOfWords(
        os.path.join("data", "desc", "test_desc"),
        os.path.join("data", "bow", "vlad_test{0}".format(num_centers)), num_centers)
    test_bow.do_bagofwords()

if __name__ == "__main__":
    main()