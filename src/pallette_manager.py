"""
This file does the initial processing of the palette colors in order to input to the neural network.
@author: Jacob Kattampilly
"""


from PIL import Image
import numpy as np
import skimage.io
import skimage.color
from glob import glob
import re
import itertools
from scipy import misc
from collections import defaultdict
from configmanager import Configs
import os

proj_imgs_path = Configs.ProcessingFolderPath
palette_dir = proj_imgs_path + 'palettes/'
images_fullsized_dir = proj_imgs_path + 'slices/'
images_3232_dir = proj_imgs_path + 'slices_resized/'

def writeLABColorsToFile(palette_dir=palette_dir, filename='LAB_palettes.txt'):
    '''
    This function writes the LAB values of the colors in the palettes in the
    palette_dir filename = path where data is written
    each line of the text contains the path of the palette followed by
    space seperated rgb values of colors in the palette
    '''
    filename = proj_imgs_path + r'/' + filename
    output_file = open(filename, 'w')
    palette_search_string = palette_dir + '*.png'
    # get each png file from the directory
    for palette_path in glob(palette_search_string):
        filename = re.search(r'(?:.*)\/(?P<flnme>.*$)', palette_path)
        output_file.write(filename.group('flnme') + ':')
        LAB_palette = get_LAB_palette_from_file(palette_path)
        output_file.write('{}'.format(LAB_palette))
        output_file.write('\n')

def get_palette_first_row(palette):
    return [list(px) for px in palette[0]]

def get_unique_palette_colors(palette_first_row):
    """
    Simply returns the unique color pixels from the palette inage
    :param palette_first_row:
    :return:
    """
    return [list(px) for px in set(tuple(px) for px in palette_first_row)]


def get_LAB_palette_from_file(palette_path):
    """
    Gets the lab values of a single palette image
    :param palette_path: path of the palette image
    :return:
    """
    lab_designer_palette = skimage.color.rgb2lab(
                                skimage.io.imread(palette_path))
    palette_first_row = get_palette_first_row(lab_designer_palette)
    return get_unique_palette_colors(palette_first_row)

def read_palette_colors_file(palettes_filename='LAB_palettes.txt',
                             rogue_palettes_filename='rogue_palettes.txt'):
    """
    This function simply reads the file created by the writeRGBColorsToFile function
    :param filename:
    :return:
    a dictionary key = filename value = list of palettes
    """
    palettes_filepath = Configs.ProcessingFolderPath + palettes_filename
    palettes_dict = defaultdict(list)
    pixel_pattern = '\[-*[0-9]+\.[0-9]+, -*[0-9]+\.[0-9]+, -*[0-9]+\.[0-9]+\]'
    float_pattern = '-*[0-9]+\.[0-9]+'

    if not os.path.exists(palettes_filepath):
        writeLABColorsToFile(palettes_filename)

    roguesfile = Configs.ProcessingFolderPath + 'file_palette_RGBmapping_rogues.txt'
    roguefilelist = [line.strip() for line in open(roguesfile)]

    with open(palettes_filepath) as f:
        for line in f:
            name, pixel_list = line.split(':')
            if name not in roguefilelist:
                pixels = re.findall(r'' + pixel_pattern, pixel_list)

                for pixel in pixels:
                    float_pixel = list(map(float, re.findall(r'' + float_pattern,
                                                             pixel)))
                    palettes_dict[name].append(float_pixel)
                palettes_dict[name] = np.array(palettes_dict[name])

    return palettes_dict

# the artist palettes dict will contain all the palettes from the lab_palettes.txt file
# if this file does not exist then it will be created from the palettes and slices
# folders respectively
artistpalettes_dict = read_palette_colors_file()


def getArtistPalette(imagename):
    """
    returns the palette for a single image from artistpalettes_dict

    :param imagename:
    :return:
    """
    return artistpalettes_dict[imagename]

def getArtistPaletteCount(imagename):
    """
    returns the count of colors in the palette for a single image from artistpalettes_dict
    :param imagename:
    :return:
    """
    return len(artistpalettes_dict[imagename])

def DisplaySIngleImageFromPath(filepath = '/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/slices/AboardColor610.png'):
    #img = Image.open(filepath)
    imgarray = misc.imread(filepath, mode='RGB')
    img = Image.fromarray(imgarray,'RGB')
    img.show()



if __name__=='__main__':
    #ResizeAllImages()
    #missingfiles()
    #writeLABColorsToFile()
    read_palette_colors_file()
    #analyzepalettecounts()
    #DisplaySIngleImageFromPath()

    #print(getArtistPalette('AboardColor610.png'))
    #print(getArtistPaletteCount('AboardColor610.png'))
