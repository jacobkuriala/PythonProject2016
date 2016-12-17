"""
This module runs all the functions in featureex on the images to extract all data
"""

import featurex
import skimage
import skimage.io as sio

dir_path = '/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/'
palette_path = dir_path + 'palettes/AbstractColor.png'
img_path = dir_path + 'slices/AbstractColor.png'


img_pixels = sio.imread(img_path)
k_means_palette_colors = kmeans(img_pixels)
designer_palette_colors = list(set(sio.imread(palette_path)))
print(palette_colors)

dataline = img_path
for i in dir(featurex):
    if i.startswith('extract_'):
        myfun = getattr(featurex, i)
        result = myfun(img_pixels, k_means_palette_colors,
                       designer_palette_colors)
        for item in result:
            dataline = dataline + ' ' + str(item)
#Instead of printing here we need to add to file or op array
print(dataline)
