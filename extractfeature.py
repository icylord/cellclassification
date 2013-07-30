__author__ = 'ShengyinWu'

import numpy as np
import DescriptorIO as desc
import os

class FeatureExtractor:
    def __init__(self, image_dir, extracted_feature_dir):
        self.image_dir = image_dir
        self.extracted_feature_dir = extracted_feature_dir

        # create tmp dir to store temp data
        if not os.path.exists(os.path.join("data", "tmp")):
            os.makedirs(os.path.join("data", "tmp"))

    def do_extraction(self):
        if not os.path.exists(self.extracted_feature_dir):
            os.makedirs(self.extracted_feature_dir)
        files_in_image_dir = os.listdir(self.image_dir)
        for f in files_in_image_dir:
            if f[-1] == "g":
                image_name = os.path.join(self.image_dir, f)
                temp_bin = os.path.join("data" , "tmp", "tmp.bin")
                colordescriptor_params = "--detector densesampling --ds_spacing 6 --ds_scales 1.6+2.4+3.2 --descriptor sift --outputFormat binary --output "
                colordescriptor_exe = os.path.join("data", "colorDescriptor", "colorDescriptor")
                command = "{0} {1} {2} {3}".format(colordescriptor_exe, image_name, colordescriptor_params, temp_bin)
                print "Extracting file {0}".format(image_name)
                os.system(command)
                keypoints, extrected_descriptors = desc.readDescriptors(temp_bin)
                extracted_feature_name = os.path.join(self.extracted_feature_dir, f[0:-4] + ".npy")
                fd = file(extracted_feature_name, "wb")
                np.save(fd, keypoints)
                np.save(fd, extrected_descriptors)

def main():
    train_feature_extractor = FeatureExtractor(os.path.join("data/images/training"), os.path.join("data/desc/training_desc"))
    train_feature_extractor.do_extraction()
    test_feature_extractor = FeatureExtractor(os.path.join("data/images/test"), os.path.join("data/desc/test_desc"))
    test_feature_extractor.do_extraction()

if __name__ == "__main__":
    main()