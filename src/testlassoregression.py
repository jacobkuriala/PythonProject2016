import lassoregression
import unittest
import pallette_manager as pm

class TestLassoRegression(unittest.TestCase):

    def test_simplelassotest(self):
        self.assertIsNone(lassoregression.test_model())

    def test_simplelassopredict(self):
        temppalette = pm.getArtistPalette('ADoorPalette.png')
        self.assertIsNone(lassoregression.production_model(temppalette))