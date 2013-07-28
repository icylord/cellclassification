__author__ = 'ShengyinWu'

import numpy as np
import os
from hkmeans import hkmeans

CLASSES_LIST=['Asterionellopsis', 'Chaetoceros', 'ciliate','Cylindrotheca', 'DactFragCeratul', 'Dactyliosolen',
                'detritus', 'Dinobryon', 'dinoflagellate', 'Ditylum', 'Euglena',
                'Guinardia',  'Licmophora',  'nanoflagellate',  'other_lt20',
                'pennate', 'Phaeocystis', 'Pleurosigma', 'Pseudonitzschia',
                'Rhizosolenia', 'Skeletonema', 'Thalassiosira']

class PerformKmeans():
    def __init__(self, words, desc_path):
        self.words = words
        self.desc_path = desc_path

    def load_data(self):
        total_descriptors = 0
        for cls in CLASSES_LIST:
            cls_path = os.path.join(self.desc_path, cls)
            all_files = os.listdir(cls_path)

            feature_length = -1
            for f in all_files:
                if f[-1] == "y":
                    npy_file = os.path.join(cls_path, f)
                    fd = file(npy_file)
                    points = np.load(fd)
                    descriptors = np.load(fd)
                    total_descriptors += descriptors.shape[0]
                    feature_length = descriptors.shape[1]

        self.descriptors = np.zeros((total_descriptors, feature_length), np.float)
        index = 0
        for cls in CLASSES_LIST:
            cls_path = os.path.join(self.desc_path, cls)
            all_files = os.listdir(cls_path)
            for f in all_files:
                if f[-1] == "y":
                    npy_file = os.path.join(cls_path, f)
                    fd = file(npy_file)
                    points = np.load(fd)
                    descriptors = np.load(fd)
                    self.descriptors[index:index + descriptors.shape[0], :] = descriptors[:,:]
                    index += descriptors.shape[0]

    def do_kmeans(self):
        norm_descriptors = np.sum(np.abs(self.descriptors)**2, axis=-1)**(1./2)
        tiled_norm_descriptors = np.tile(norm_descriptors, (self.descriptors.shape[1], 1))
        self.descriptors = self.descriptors / tiled_norm_descriptors.T
        self.kmeans_centers = hkmeans(self.descriptors, self.words)

    def save_trained_words(self):
        words_dir = os.path.join("data", "words")
        if not os.path.exists(words_dir):
            os.makedirs(words_dir)
        saved_words_file = os.path.join(words_dir, "{0}.npy".format(self.words))
        fd = file(saved_words_file, "wb")
        np.save(fd, self.kmeans_centers)

if __name__ == "__main__":
    perform_kmeans = PerformKmeans(256, os.path.join("data", "desc", "training_desc"))
    perform_kmeans.load_data()
    perform_kmeans.do_kmeans()
    perform_kmeans.save_trained_words()