"""This module contains functions to read/write features to a text file"""

import re
from collections import defaultdict
import clustermanager as cm
import featurex as fx
import pallette_manager as pm
from configmanager import Configs


directory_path = Configs().ProcessingFolderPath
image_path = directory_path + 'slices/'
features_fname = directory_path + 'features.txt'


def write_features(filename=features_fname):
    """
    Write features to file with name filename.

    Args:
    filename: full path for feature file
    """
    palette_dict = pm.read_palette_colors_file()
    feature_file = open(filename, 'w')

    for fname, palette_pixels in palette_dict.items():
        feature_file.write(fname + '_kmeans:')
        kmeans_palette_features = fx.feature_extraction(palette_pixels,
                                                        cm.findimagekmeans(fname))
        feature_file.write('{}'.format(kmeans_palette_features))

        feature_file.write('\n' + fname + '_minibatchmeans:')
        minibatchmeans_palette_features = fx.feature_extraction(palette_pixels,
                                                                cm.findimageminibatchmeans(fname))

        feature_file.write('{}'.format(minibatchmeans_palette_features))

        feature_file.write('\n' + fname + '_randmeans:')
        randmeans_palette_features = fx.feature_extraction(palette_pixels,
                                                           cm.findimagerandommeans(fname))
        feature_file.write('{}\n'.format(randmeans_palette_features))

        feature_file.write(fname + '_designer:')

        designer_palette_features = fx.feature_extraction(palette_pixels,
                                                          palette_pixels,
                                                          equal=True)
        feature_file.write('{}\n'.format(designer_palette_features))


def read_features(filename=features_fname):
    """
    Read features from file

    Args:
    features_fname: stores files in this filename

    Returns:
    A dictionary of image names as keys and list of features as values
    """

    features_dict = defaultdict(list)
    float_pattern = r'-*[0-9]+\.[0-9]+'
    with open(filename) as file:
        for line in file:
            name, feature_list = line.split(':')
            features = re.findall(r'' + float_pattern, feature_list)
            float_features = [float(feature) for feature in features]
            features_dict[name] = float_features

    return features_dict

if __name__ == '__main__':
    write_features()
