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
        files_in_dir = os.listdir(self.desc_path)
        word_file = os.path.join(TOP_DIR, "data", "words", "{0}.npy".format(word))
        fd = file(word_file, "rb")
        codebook = np.load(fd)
        fd.close()
        for f in files_in_dir:
            print f
            if f[-1] == "y":
                npy_file = os.path.join(self.desc_path, f)
                # may be some files broken
                if os.path.getsize(npy_file) < 300:
                    continue
                fd = file(npy_file)
                keypoints = np.load(fd)
                data = np.load(fd)
                png_file = os.path.join(self.png_path, f[0:-4] + ".png")
                loaded_image = Image.open(png_file)
                image_width = loaded_image.size[0]
                image_height = loaded_image.size[1]
                xs = keypoints[:, 0]
                ys = keypoints[:, 1]

                data_norm = np.sum(np.abs(data)**2, axis=-1)**(1./2) + 1e-12
                tile_data_norm = np.tile(data_norm, (data.shape[1], 1))
                data = data / tile_data_norm.T
                fd_feature = file(os.path.join(self.bow_path, f), "wb")
                fea = vl_coding(data, codebook, 3, 5, image_width, image_height, xs, ys)
                np.save(fd_feature, fea)

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