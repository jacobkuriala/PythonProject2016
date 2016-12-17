import scorer
import unittest
import os
from configmanager import Configs

class TestScorer(unittest.TestCase):

    def test_readscoresfromfile(self):
        """
        Also tests writing by deleting the files if they exist first
        :return:
        """
        if os.path.isfile(Configs.ProcessingFolderPath + scorer.scorefilename):
            self.removefile(scorer.scorefilename)
        scores = scorer.readscoresfromfile()
        self.assertIsNotNone(scores)
        print(scores)
        self.assertGreater(len(scores), 0)

        score = scorer.getscore('ADoorPalette.png_minibatchmeans')
        self.assertIsNotNone(score)


    def removefile(self,filename):
        testfilepath = Configs.ProcessingFolderPath + filename
        if os.path.isfile(testfilepath):
            print('deleting', testfilepath)
            os.remove(testfilepath)

    def test_deserialize_invalid(self):
        """
        Testing deserializing an invalid file
        :return:
        """
        des_file = scorer.deserialize('someinvalidfile')
        self.assertEqual(0, len(des_file))

        file = open(Configs.ProcessingFolderPath + r'testfile', 'w')
        file.writelines('123')
        file.close()
        des_file = scorer.deserialize(Configs.ProcessingFolderPath + r'testfile')
        self.assertEqual(0, len(des_file))

    def tearDown(self):
        testfilepath = Configs.ProcessingFolderPath + r'testfile'
        if os.path.isfile(testfilepath):
            print('deleting', testfilepath)
            os.remove(testfilepath)
