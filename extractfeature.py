__author__ = 'ShengyinWu'

import os
import numpy as np
import DescriptorIO

class FeatureExtractor:
    def __init__(self, image_dir, extracted_descriptor_dir):
        self.image_dir = image_dir
        self.extracted_descriptor_dir = extracted_descriptor_dir

        # create tmp dir to store temp data
        if not os.path.exists(os.path.join("data", "tmp")):
            os.makedirs(os.path.join("data", "tmp"))

    def doExtraction(self):
        if not os.path.exists(self.extracted_descriptor_dir):
            os.makedirs(self.extracted_descriptor_dir)
        files_in_dir = os.listdir(self.image_dir)
        for f in files_in_dir:
            if f[-1] == "g":
                image_name = os.path.join(self.image_dir, f)
                temp_bin = os.path.join("data" , "tmp", "tmp.bin")
                colordescriptor_params = "--detector densesampling --ds_spacing 6 --ds_scales 1.6+2.4+3.2 --descriptor sift --outputFormat binary --output "
                colordescriptor_exe = os.path.join("data", "colorDescriptor", "colorDescriptor")
                exec_command = "{0} {1} {2} {3}".format(colordescriptor_exe, image_name, colordescriptor_params, temp_bin)
                os.system(exec_command)
                keypoints, descriptors = DescriptorIO.readDescriptors(temp_bin)
                extracted_feature_name = os.path.join(self.extracted_descriptor_dir, f[0:-4] + ".npy")
                fd = file(extracted_feature_name, "wb")
                np.save(fd, descriptors)
                np.save(fd, keypoints)

def main():
    train_feature_extractor = FeatureExtractor(os.path.join("data/images/training"), os.path.join("data/desc/training_desc"))
    train_feature_extractor.doExtraction()
    test_feature_extractor = FeatureExtractor(os.path.join("data/images/test"), os.path.join("data/desc/test_desc"))
    test_feature_extractor.doExtraction()

if __name__ == "__main__":
    main()