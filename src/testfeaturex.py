import featurex as fx
import featureexhelpers as fxh
import numpy as np
import unittest
import skimage.io as sio
import clustermanager
from configmanager import Configs


class TestFeatureExtraction(unittest.TestCase):
    """
    This class is used to test all the functions in featureex.py
    """
    def setUp(self):
        testimg = 'ADoorHues_10.png'
        imagepath = Configs().ProcessingFolderPath + 'slices/' + testimg
        self.testImage = sio.imread(imagepath)[0]
        self.kmeansfortestImage = clustermanager.findimagekmeans(testimg)
        self.test_array = np.asarray([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1],
                                                               [1, 1, 1]]])
        self.test_palette = np.array([[1, 2, 3], [4, 5, 6], [8, 9, 10],
                                      [13, 14, 15], [19, 20, 21]])

    def test_RGB_mean(self):
        #testing 2-d image with all ones
        self.assertEqual(fx.extract_RGBmean(self.test_array), (1.0, 1.0, 1.0))

    def test_Lcov(self):
        pass

    def test_Scov(self):
        pass

    def test_soft_recoloring_error(self):
        pass

    def test_min_color_dist(self):
        self.assertEqual(fx.extract_min_color_dist(self.test_palette),
                         fxh.calculate_euclid_dist(np.array(
                             self.test_palette[0]),
                                                   np.array(self.test_palette[1])))

    def test_max_color_dist(self):
        self.assertEqual(fx.extract_max_color_dist(self.test_palette),
                         fxh.calculate_euclid_dist(self.test_palette[0],
                                                   self.test_palette[4]))

    def test_mean_color_dist(self):
        pass

    def test_extract_Lcov_with_image(self):
        print('run test_extract_Lcov_with_images')
        print(fx.extract_Lcov(self.testImage, self.kmeansfortestImage))

    def test_extract_Scov_with_image(self):
        print('run test_extract_Scov_with_images')
        print(fx.extract_Scov(self.testImage, self.kmeansfortestImage))

    def test_extract_soft_recoloring_error_with_image(self):
        print('run test_extract_soft_recoloring_error_images')
        print(fx.extract_soft_recoloring_error(self.testImage,
                                               self.kmeansfortestImage))

if __name__ == '__main__':
    unittest.main()
