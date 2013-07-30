from extractfeature import *
from mykmeans import *
from mybow import *
import trainandeval

def main():
	# #1
 #    print "Start Extracting Features"
 #    cls = ""
 #    train_feature_extractor = FeatureExtractor(os.path.join("data/images/training/{0}".format(cls)),
 #                                                os.path.join("data/desc/training_desc/{0}".format(cls)))
 #    train_feature_extractor.do_extraction()

 #    test_feature_extractor = FeatureExtractor(os.path.join("data/images/test/{0}".format(cls)),
 #                                                os.path.join("data/desc/test_desc/{0}".format(cls)))

 #    #2
 #    print "Start Kmeans"
 #    word = 256
 #    perform_kmeans = PerformKmeans(word, os.path.join("data", "desc", "training_desc"))
 #    perform_kmeans.load_data()
 #    perform_kmeans.do_kmeans()
 #    perform_kmeans.save_trained_words()

 #    #3
 #    print "Coding"
 #    train_bag_of_visual_words = BagOfVisualWords(
 #        os.path.join("data", "images", "training"),
 #        os.path.join("data", "desc", "training_desc"),
 #        os.path.join("data", "bow", "vlad_train{0}".format(word)), word)
 #    train_bag_of_visual_words.do_bag_of_visual_words()
 #    test_bag_of_visual_words = BagOfVisualWords(
 #        os.path.join("data", "images", "test"),
 #        os.path.join("data", "desc", "test_desc"),
 #        os.path.join("data", "bow", "vlad_test{0}".format(word)), word)
 #    test_bag_of_visual_words.do_bag_of_visual_words()

    #4 train and evaluation
    print "train and evaluation"
    trainandeval.main()

if __name__ == '__main__':
	main()

