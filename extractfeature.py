__author__ = 'ShengyinWu'

import numpy as np
import DescriptorIO as desc
import os
import Image

class FeatureExtractor:
    def __init__(self, image_dir, extracted_feature_dir):
        self.image_dir = image_dir
        self.extracted_feature_dir = extracted_feature_dir

        # create tmp dir to store temp data
        if not os.path.exists(os.path.join("tmp")):
            os.makedirs(os.path.join("tmp"))

    def do_extraction(self):
        if not os.path.exists(self.extracted_feature_dir):
            os.makedirs(self.extracted_feature_dir)
        files_in_image_dir = os.listdir(self.image_dir)
        for f in files_in_image_dir:
            if f[-1] == "g":
                image_name = os.path.join(self.image_dir, f)

                loaded_image = Image.open(image_name)
                image_size = loaded_image.size
                scale_width = image_size[0] / 150.0
                scale_height = image_size[1] / 150.0
                maxscale = max(scale_width, scale_height)
                resize_image = loaded_image.resize((int(image_size[0] / maxscale), int(image_size[1] / maxscale)))

                png_path = os.path.join(self.image_dir, f[0:-4])
                png_fullpath = "{0}.png".format(png_path)
                resize_image.save(png_fullpath)

                temp_bin = os.path.join("tmp", "tmp.bin")
                colordescriptor_params = "--detector densesampling --ds_spacing 6 --ds_scales 1.6+2.4+3.2 --descriptor sift --outputFormat binary --output "
                colordescriptor_exe = os.path.join("colorDescriptor", "colorDescriptor")
                command = "{0} {1} {2} {3}".format(colordescriptor_exe, png_fullpath, colordescriptor_params, temp_bin)

                print "Extracting file {0}".format(image_name)
                os.system(command)
                keypoints, extrected_descriptors = desc.readDescriptors(temp_bin)
                extracted_feature_name = os.path.join(self.extracted_feature_dir, f[0:-4] + ".npy")
                fd = file(extracted_feature_name, "wb")
                np.save(fd, keypoints)
                np.save(fd, extrected_descriptors)

if __name__ == "__main__":
    #for cls in CLASSES_LIST:
        cls = ""
        train_feature_extractor = FeatureExtractor(os.path.join(TOP_DIR, "data/images/training/{0}".format(cls)),
                                                   os.path.join(TOP_DIR, "data/desc/training_desc/{0}".format(cls)))
        train_feature_extractor.do_extraction()

        test_feature_extractor = FeatureExtractor(os.path.join(TOP_DIR, "data/images/test/{0}".format(cls)),
                                                  os.path.join(TOP_DIR, "data/desc/test_desc/{0}".format(cls)))
        test_feature_extractor.do_extraction()