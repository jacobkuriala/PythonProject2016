import pallette_manager as pm
import clustermanager as cm
import featurex as fx
from collections import defaultdict
from configmanager import Configs
import re

#dir_path = '/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/'
dir_path = Configs.ProcessingFolderPath
img_path = dir_path + 'slices/'
features_filename = dir_path + 'features.txt'
# 'featurespixelcoverage.txt'


def feature_extraction(designer_palette, generated_palette, equal=False):
    features = []
    features.append(fx.extract_Lcov(designer_palette, generated_palette))
    features.append(fx.extract_Scov(designer_palette, generated_palette))
    features.append(fx.extract_soft_recoloring_error(designer_palette,
                                                     generated_palette, equal))
    features.append(fx.extract_min_color_dist(generated_palette))
    features.append(fx.extract_max_color_dist(generated_palette))
    features.append(fx.extract_mean_color_dist(generated_palette))


    return features

def write_features(features_filename=features_filename):
    palette_dict = pm.read_palette_colors_file()
    feature_file = open(features_filename, 'w')

    for filename, palette_pixels in palette_dict.items():

        feature_file.write(filename + '_kmeans:')
        kmeans_palette_features = feature_extraction(palette_pixels,
                                                     cm.findimagekmeans(filename))
        feature_file.write('{}'.format(kmeans_palette_features))

        feature_file.write('\n' + filename + '_minibatchmeans:')
        minibatchmeans_palette_features = feature_extraction(palette_pixels,
                                                             cm.findimageminibatchmeans(filename))
        feature_file.write('{}'.format(minibatchmeans_palette_features))

        feature_file.write('\n' + filename + '_randmeans:')
        randmeans_palette_features = feature_extraction(palette_pixels,
                                                        cm.findimagerandommeans(filename))
        feature_file.write('{}\n'.format(randmeans_palette_features))

        feature_file.write(filename + '_designer:')

        designer_palette_features = feature_extraction(palette_pixels,
                                                       palette_pixels,
                                                       equal=True)
        feature_file.write('{}\n'.format(designer_palette_features))


def read_features(f_filename=features_filename):
    features_dict = defaultdict(list)
    float_pattern = '-*[0-9]+\.[0-9]+'
    with open(f_filename) as f:
        for line in f:
            name, feature_list = line.split(':')
            features = re.findall(r'' + float_pattern, feature_list)
            float_features = list(map(float, features))
            features_dict[name] = float_features

    return features_dict

if __name__ == '__main__':
    write_features()
