import pallette_manager as pm
import unittest
from configmanager import Configs
import os

class TestPaletteManager(unittest.TestCase):

    def test_writeLABColorsToFile(self):
        filename = r'LAB_palettes.txt'
        self.removefile(filename)
        pm.writeLABColorsToFile(filename=filename)
        testfilepath = Configs.ProcessingFolderPath + filename
        # print(testfilepath)
        if not os.path.isfile(testfilepath):
            self.fail(msg='LAB file not created')

    def test_read_palette_colors_file(self):
        filename = r'LAB_palettes.txt'
        tempcheck = pm.read_palette_colors_file(filename)
        self.assertIsNotNone(tempcheck)
        self.assertGreater(len(tempcheck), 0)
        # test simple functions with read
        artistpalette = pm.getArtistPalette('ADoorPalette.png')
        self.assertIsNotNone(artistpalette)
        artistpalettecount = pm.getArtistPaletteCount('ADoorPalette.png')
        self.assertGreater(artistpalettecount, 0)
        #print(artistpalettecount)

    def removefile(self,filename):
        testfilepath = Configs.ProcessingFolderPath + filename
        if os.path.isfile(testfilepath):
            print('deleting', testfilepath)
            os.remove(testfilepath)

    def tearDown(self):
        """
        filenamestodelete = [r'LAB_palettes.txt']
        for filename in filenamestodelete:
            testfilepath = Configs.ProcessingFolderPath + filename
            if os.path.isfile(testfilepath):
                self.removefile(filename)
        """
