import featureexhelpers as fxh
import numpy as np
import math
import unittest

class TestFeatureExHelper(unittest.TestCase):
    """
    This class is used to test all the functions in the featureexhelpers
    """
    def test_mean(self):
        l = [1, 2, 3, 4, 5]
        self.assertEqual(fxh.calculate_mean(l), 3)

    def test_euclid_dist(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        self.assertEqual(fxh.calculate_euclid_dist(a, b), math.sqrt(9*3))

    def test_upc(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        c = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6], [4, 5, 6]])
        self.assertEqual(fxh.calculate_upc(a, b, c), 0.2)

    def test_distances(self):
        pass

    def test_formatter(self):
        pass
