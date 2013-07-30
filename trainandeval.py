__author__ = 'ShengyinWu'

import sys
import os
import numpy as np
from sklearn import svm
import evaluate_results

class_dict = {'homogeneous':1, 'coarse_speckled':2, 'fine_speckled':3, 'nucleolar':4, 'centromere':5, 'cytoplasmatic':6}

class_list=['homogeneous', 'coarse_speckled', 'fine_speckled', 'nucleolar', 'centromere', 'cytoplasmatic']

# evaluation and return confusion matrix
def evaluation(data, cls, dec):
    predicted = dec.predict(data)
    catergory = []
    aps = []
    for cls in class_list:
        v = class_dict[cls]
        class_index = (cls == v)
        catergory.append(cls)
        mask = (cls[class_index] == predicted[class_index])
        if class_index.sum(axis=0) == 0:
            continue
        aps.append( float(sum(mask)) / float(class_index.sum(axis=0)))
        print "{0} {1}".format(cls, float(sum(mask)) / float(class_index.sum(axis=0)))
    mask = (cls == predicted)
    print "Total {0}".format(float(sum(mask)) / float(cls.shape[0]))
    sys.stdout.flush()

    confusion_matrix = np.zeros((6, 6), np.float)
    for cls in class_list:
        v = class_dict[cls]
        mask = (cls == v)
        cater_vec = predicted[mask]
        for i in range(cater_vec.shape[0]):
            confusion_matrix[int(v)-1, int(cater_vec[i])-1] += 1

    for i in range(6):
        confusion_matrix[i, :] = confusion_matrix[i, :] / np.sum(confusion_matrix[i, :])
    print confusion_matrix * 100
    return confusion_matrix

def main():
    word = 256
    fea_path = 128
    bow_train_dir = "data/bow/vlad_train{0}".format(word)
    bow_test_dir = "data/bow/vlad_test{0}".format(word)

    files = os.listdir(bow_train_dir)
    n_trainsample = len(files)
    print "n_trainsample = {0}".format(n_trainsample)

    # load ground true labels
    csv_file = evaluate_results.read_csv("data/images/training/gt_training.csv")

    train_data = np.empty((n_trainsample, fea_path*word), np.float64)
    train_cls = np.empty(n_trainsample, np.int)

    ind = 0
    for item in csv_file:
        f = "%03d.npy" %(int(item['id']))
        bow_file = os.path.join(bow_train_dir, f)
        fd = file(bow_file, 'rb')
        bow_feature = np.load(fd)
        train_data[ind, :] = bow_feature
        train_cls[ind] = int(class_dict[item["pattern"]])
        ind += 1

    clf = svm.LinearSVC(C=1)
    dec = clf.fit(train_data, train_cls)
    evaluation(train_data, train_cls, dec)

    ##### TEST #####
    files = os.listdir(bow_test_dir)
    n_testsample = len(files)
    print "n_testsample = {0}".format(n_testsample)

    # load ground true labels
    csv_file = evaluate_results.read_csv("data/images/test/gt_test.csv")

    test_data = np.empty((n_testsample, fea_path*word), np.float64)
    test_cls = np.empty(n_testsample, np.int)

    ind = 0
    for item in csv_file:
        f = "%03d.npy" %(int(item['id']))
        bow_file = os.path.join(bow_test_dir, f)
        fd = file(bow_file, 'rb')
        bow_feature = np.load(fd)
        test_data[ind, :] = bow_feature
        test_cls[ind] = int(class_dict[item["pattern"]])
        ind += 1
    conf_matrix = evaluation(test_data, test_cls, dec)

if __name__ == "__main__":
    main()


