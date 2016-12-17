"""
Module contains Class Configs which has the main data path used for data
"""


class Configs(object):
    """
    Maintains the high level configuration like:
    processing folder path which is the location where all
    the data is located
    """
    # the processing folder is meant to be the main folder where all the assets
    # will exist there are certain assumptions that
    #  are made about the ProcessingFolderPath
    # !!Add Assumptions!!
    # There is a slices folder present with image assets
    # There is a palettes folder present with palettes matching
    #  the images in theslices
    # Intermediate files get added to this folder
    # ProcessingFolderPath = \
    #   r'/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/'
    # ProcessingFolderPath = \
    #   r'/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/'
    ProcessingFolderPath = r'Data/sample_data/'
