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
    pred = dec.predict(data)
    caters = []
    aps = []
    for k in class_list:
        v = class_dict[k]
        clsind = (cls == v)
        caters.append(k)
        mask = (cls[clsind] == pred[clsind])
        if clsind.sum(axis=0) == 0:
            continue
        aps.append( float(sum(mask)) / float(clsind.sum(axis=0)))
        print "{0} {1}".format(k, float(sum(mask)) / float(clsind.sum(axis=0)))
    mask = (cls == pred)
    print "Average AP: {0}".format(float(sum(mask)) / float(cls.shape[0]))
    sys.stdout.flush()

    conf_matrix = np.zeros((6, 6), np.float)
    for k in class_list:
        v = class_dict[k]
        mask = (cls == v)
        cater_vec = pred[mask]
        for i in range(cater_vec.shape[0]):
            conf_matrix[int(v)-1, int(cater_vec[i])-1] += 1

    for i in range(6):
        conf_matrix[i, :] = conf_matrix[i, :] / np.sum(conf_matrix[i, :])
    return conf_matrix

def main():
    word = 256
    fea_path = 128
    bow_train_path = "data/bow/vlad_train{0}".format(word)
    bow_test_path = "data/bow/vlad_test{0}".format(word)

    files = os.listdir(bow_train_path)
    n_trainsample = len(files)
    print "n_trainsample = {0}".format(n_trainsample)

    # load ground true labels
    csv_file = evaluate_results.read_csv("data/images/training/gt_training.csv")

    train_data = np.empty((n_trainsample, fea_path*word), np.float64)
    train_cls = np.empty(n_trainsample, np.int)

    ind = 0
    for item in csv_file:
        f = "%03d.npy" %(int(item['id']))
        npyname = os.path.join(bow_train_path, f)
        fd = file(npyname, 'rb')
        bowfeature = np.load(fd)
        train_data[ind, :] = bowfeature
        train_cls[ind] = int(class_dict[item["pattern"]])
        ind += 1

    clf = svm.LinearSVC(C=1)
    dec = clf.fit(train_data, train_cls)
    evaluation(train_data, train_cls, dec)

    ##### TEST #####
    files = os.listdir(bow_test_path)
    n_testsample = len(files)
    print "n_testsample = {0}".format(n_testsample)

    # load ground true labels
    csv_file = evaluate_results.read_csv("data/images/test/gt_test.csv")

    test_data = np.empty((n_testsample, fea_path*word), np.float64)
    test_cls = np.empty(n_testsample, np.int)

    ind = 0
    for item in csv_file:
        f = "%03d.npy" %(int(item['id']))
        npyname = os.path.join(bow_test_path, f)
        fd = file(npyname, 'rb')
        bowfeature = np.load(fd)
        test_data[ind, :] = bowfeature
        test_cls[ind] = int(class_dict[item["pattern"]])
        ind += 1
    confusioni_matrix = evaluation(test_data, test_cls, dec)
    print confusioni_matrix * 100

if __name__ == "__main__":
    main()


