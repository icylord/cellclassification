__author__ = 'ShengyinWu'

import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans

class_list=['']

class PerformKmeans():
    def __init__(self, num_centers, extracted_descriptor_dir):
        self.num_centers = num_centers
        self.descriptor_dir = extracted_descriptor_dir

    def loadData(self):
        total_descriptors = 0
        for cls in class_list:
            cls_path = os.path.join(self.descriptor_dir, cls)
            files_in_dir = os.listdir(cls_path)

            feature_dim = -1
            for files in files_in_dir:
                if files[-1] == "y":
                    descriptor_file = os.path.join(cls_path, files)
                    fd = file(descriptor_file)
                    descriptors = np.load(fd)
                    total_descriptors += descriptors.shape[0]
                    feature_dim = descriptors.shape[1]

        self.descriptors = np.zeros((total_descriptors, feature_dim), np.float)

        index = 0
        for cls in class_list:
            cls_path = os.path.join(self.descriptor_dir, cls)
            files_in_dir = os.listdir(cls_path)
            for files in files_in_dir:
                if files[-1] == "y":
                    descriptor_file = os.path.join(cls_path, files)
                    fd = file(descriptor_file)
                    descriptors = np.load(fd)
                    self.descriptors[index:index + descriptors.shape[0], :] = descriptors[:,:]
                    index += descriptors.shape[0]

    def doKmeans(self):
        norm_descriptors = np.sum(np.abs(self.descriptors)**2, axis=-1)**(1./2)
        tiled_norm_descriptors = np.tile(norm_descriptors, (self.descriptors.shape[1], 1))
        self.descriptors = self.descriptors / tiled_norm_descriptors.T
        km = MiniBatchKMeans(n_clusters = self.num_centers, init='k-means++', batch_size = self.num_centers * 8,
                             verbose = 1, compute_labels = False)
        self.codebook = km.fit(self.descriptors)

    def saveCodebook(self):
        codebook_dir = os.path.join("data", "words")
        if not os.path.exists(codebook_dir):
            os.makedirs(codebook_dir)
        saved_codebook_file = os.path.join(codebook_dir, "{0}.npy".format(self.num_centers))
        fd = file(saved_codebook_file, "wb")
        np.save(fd, self.codebook)

def main():
    perform_kmeans = PerformKmeans(256, os.path.join("data", "desc", "training_desc"))
    perform_kmeans.loadData()
    perform_kmeans.doKmeans()
    perform_kmeans.saveCodebook()

if __name__ == "__main__":
    main()
