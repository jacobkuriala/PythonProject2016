import unittest
from configmanager import Configs
import featureio
import featurex
import os


class TestFeatureIO(unittest.TestCase):
    """
    Class used to test clustermanager
    """
    def setUp(self):
        featuresfilename = featureio.features_fname
        if os.path.isfile(featuresfilename):
            print('deleting', featuresfilename)
            os.remove(featuresfilename)

    def test_read_features(self):
        features = featureio.read_features()
        self.assertIsNotNone(features)
        self.assertGreater(len(features),0)


    def tearDown(self):
        pass

