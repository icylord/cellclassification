__author__ = 'ShengyinWu'

import numpy as np
import os
import Image

from coding import *

class BagOfVisualWords:
    def __init__(self, png_path, desc_path, bow_path, word):
        self.png_path = png_path
        self.desc_path = desc_path
        self.bow_path = bow_path
        self.word = word
        self.coding_and_pooling = CodingAndPooling()

    def do_bag_of_visual_words(self):
        if not os.path.exists(self.bow_path):
            os.makedirs(self.bow_path)
        files = os.listdir(self.desc_path)
        n_sample = len(files)
        wordstr = os.path.join("data", "words", "{0}.npy".format(word))
        fd = file(wordstr, "rb")
        words = np.load(fd)
        fd.close()
        i = 0
        for f in files:
            print f
            if f[-1] == "y":
                npyname = os.path.join(self.desc_path, f)
                fd = file(npyname)
                points = np.load(fd)
                data = np.load(fd)
                pngname = os.path.join(self.png_path, f[0:-4] + ".png")
                im = Image.open(pngname)
                WIDTH = im.size[0]
                HEIGHT = im.size[1]
                xs = points[:, 0]
                ys = points[:, 1]

                n = np.sum(np.abs(data)**2, axis=-1)**(1./2) + 1e-12
                nn = np.tile(n, (data.shape[1], 1))
                data = data / nn.T
                fd_fea = file(os.path.join(self.bow_path, f), "wb")
                fea = vlad_coding(data, words, WIDTH, HEIGHT, xs, ys)
                np.save(fd_fea, fea)

if __name__ == "__main__":
    word = 256
    coding_and_pooling = CodingAndPooling()
    train_bag_of_visual_words = BagOfVisualWords(
        os.path.join("data", "images", "training"),
        os.path.join("data", "desc", "training_desc"),
        os.path.join("data", "bow", "m_hv_train{0}".format(word)), word)
    test_bag_of_visual_words = BagOfVisualWords(
        os.path.join("data", "images", "test"),
        os.path.join("data", "desc", "test_desc"),
        os.path.join("data", "bow", "m_hv_test{0}".format(word)), word)