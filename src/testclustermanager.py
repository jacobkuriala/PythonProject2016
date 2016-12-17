import unittest
from configmanager import Configs
import clustermanager as clusterm
import os


class TestClusterManager(unittest.TestCase):
    """
    Class used to test clustermanager
    """
    def setUp(self):
        meanfilepostfixes = ['KMeans','MiniBatchKMeans','Random']
        processingfolderpath = Configs.ProcessingFolderPath

        for meanfile in meanfilepostfixes:
            meansfilepath = processingfolderpath + 'means_' + meanfile
            if os.path.isfile(meansfilepath):
                print('deleting', meansfilepath)
                os.remove(meansfilepath)

    def test_findmeansfromlist(self):
        self.assertIsNone(clusterm.findimagekmeans(r'non_existent'))
        self.assertIsNotNone(clusterm.findimagekmeans(r'ADoorPalette.png'))
        self.assertIsNotNone(clusterm.findimageminibatchmeans(r'ADoorPalette.png'))
        self.assertIsNotNone(clusterm.findimagerandommeans(r'ADoorPalette.png'))

    def test_deserialize_invalid(self):
        """
        Testing deserializing an invalid file
        :return:
        """
        des_file = clusterm.deserialize('someinvalidfile')
        self.assertEqual(0, len(des_file))

        file = open(Configs.ProcessingFolderPath + r'testfile', 'w')
        file.writelines('123')
        file.close()
        des_file = clusterm.deserialize(Configs.ProcessingFolderPath + r'testfile')
        self.assertEqual(0, len(des_file))

    def tearDown(self):
        testfilepath = Configs.ProcessingFolderPath + r'testfile'
        if os.path.isfile(testfilepath):
            print('deleting', testfilepath)
            os.remove(testfilepath)

