import featurex as fx
import featureexhelpers as fxh
import numpy as np
import unittest
import skimage.io as sio
import clustermanager


class TestFeatureExtraction(unittest.TestCase):
    """
    This class is used to test all the functions in featureex.py
    """
    def setUp(self):
        testimg = 'ADoorColor_1.png'
        imagepath = '/home/jacob/PycharmProjects/Chameleon/images/slices/' + testimg
        self.testImage = sio.imread(imagepath)
        self.kmeansfortestImage = clustermanager.findimagekmeans(testimg, 5)
        #print(self.kmeansfortestImage)

    def test_RGB_mean(self):
        #testing 2-d image with all ones
        testarray = np.asarray([[[1, 1, 1], [1, 1, 1]],
                                [[1, 1, 1], [1, 1, 1]]])
        self.assertEqual(fx.extract_RGBmean(testarray), (1.0, 1.0, 1.0))

    def test_Lcov(self):
        pass

    def test_Scov(self):
        pass

    def test_soft_recoloring_error(self):
        pass

    def test_min_color_dist(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [8, 9, 10], [13, 14, 15],
                     [19, 20, 21]])
        self.assertEqual(fx.extract_min_color_dist(a),
                         fxh.calculate_euclid_dist(np.array(a[0]),
                                                   np.array(a[1])))

    def test_max_color_dist(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [8, 9, 10], [13, 14, 15],
                     [19, 20, 21]])
        self.assertEqual(fx.extract_max_color_dist(a),
                         fxh.calculate_euclid_dist(a[0], a[4]))

    def test_mean_color_dist(self):
        pass


    def test_extract_Lcov_with_image(self):
        print('run test_extract_Lcov_with_images')
        print(fx.extract_Lcov(self.testImage,self.kmeansfortestImage))

    def test_extract_Scov_with_image(self):
        print('run test_extract_Scov_with_images')
        print(fx.extract_Scov(self.testImage,self.kmeansfortestImage))

    def test_extract_soft_recoloring_error_with_image(self):
        print('run test_extract_soft_recoloring_error_images')
        print(fx.extract_soft_recoloring_error(self.testImage,self.kmeansfortestImage))
