# -*- coding: utf-8 -*-
'''
Created on Jul 21, 2015

@author: Polina

This file is designed to navigate through a directory and to generate
cluster points using k-clustering. The resulting color cluster points are
obtained as sRGB colors, converted to LAB color format, and stored within a
file opened for writing in the following format:
---
['AboardColor610.png', array([[ 54.64264559, -14.60460422, -10.23700763],
       [ 87.08866725,  -5.26366856,   2.27897359],
       [ 82.03646435, -18.22094808,  -4.72795878],
       [ 74.03191909,  -1.56897844,  -1.0575236 ],
       [ 73.69373218, -27.36947636,  -9.59857676]])]
['Abovetheclouds610.png', array([[ 65.80495582,   8.73096919,  17.21941252],
       [ 14.58061212,   3.64430622,  -7.30093801],
       [ 84.84347125,   3.16715576,  49.80796701],
       [ 37.08751183,   9.70655739,  -1.09391897],
       [ 66.36912056,  34.28291538,  63.57886745]])]
---
Notes: Color is LAB format. This example is for k=5 clustering.
http://scikit-learn.org/stable/modules/clustering.html#clustering
---
At the conclusion, the list is then serialized so that it may be
deserialized and python will recognize each elements of the file as
lists of [(name), (array of colors)]

Requires a folder named "slices" in the current directory with at least
one jpg/png image.
'''
import sys

#sys.path.append("/usr/local/lib/python3.4/dist-packages")
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor, AdobeRGBColor
import math
import numpy
import os
import pickle
import scipy
import os.path

# old
# import scipy.cluster
import scipy.misc
# new
from sklearn.cluster import *

# from sklearn import metrics
# from sklearn.cluster import KMeans
# from sklearn.datasets import load_digits
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import scale

numpy.random.seed(42)


def euclidean_distance(point1, point2):
    '''
    Used for calculating the distance between two points. Both
    points must have the same number of dimensions.

    Args:
        point1: A list of points
        point2: Another list of points
    Returns:
        None: if point1 and point2 are not equal length
        result: A single float representative of the euclidan distance
    Raises:
        TypeError: if either parameter is not iterable.

    >>> euclidean_distance([1], [2,3])
    >>> euclidean_distance([1,2], [1,2])
    0.0
    >>> euclidean_distance([1,2], [2,3])
    1.4142135623730951
   * >>> euclidean_distance((1,2,4), (9,2,3))
    8.06225774829855
    '''
    if len(point1) != len(point2):
        return None

    distance = 0

    # distance += (x2 - x1)^2 for each x2, x1 in points2, points1
    for value in range(len(point1)):
        try:
            distance += (point2[value] - point1[value]) ** 2
        except:
            print('Invalid point', point1, point2)
    return math.sqrt(distance)


from numpy import bincount


def find_clusters_in_dir(training_clusters, original_images_path, overwrite,meancount=5):
    """
    Checks a directory. If it is a directory, images from the folder 'slices'
    are opened one at a time. Each image is clustered, and its resulting
    cluster points are added to a list.

    Args:
        textfile_read: A file that is open for reading
        textfile_app: A file that is open for appending to the end
        original_images_path: A path that may be used to acess the images
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    """
    # clusterfilename ="backend/clusters.txt"
    # cluster_list = []
    # lines=[]
    # if overwrite or not os.path.isfile(clusterfilename):
    #     clusterfile = open("backend/clusters.txt", "w")
    # else:
    #     clusterfile = open("backend/clusters.txt", "a")
    #     try:
    #         lines = clusterfile.readlines()
    #     except:
    #         clusterfile = open("backend/clusters.txt", "w")

    skipflag = False
    cluster_list = training_clusters
    isweighted = 1
    # print('cluster_list', cluster_list[0][1])
    # if len(cluster_list[0]) !=3:
    #     isweighted=1
    print('isweighted', isweighted)
    # print('Num clusters currently in cluster file', len(lines))
    print('Num images in folder', len(os.listdir(original_images_path)))
    print('Num images in serialized cluster file', len(training_clusters))
    count_images = 0
    for imagename in os.listdir(original_images_path):
        try:
            skipflag = False
            count_images = count_images + 1
            # Skip if image is already in file
            for name, colors in training_clusters:
                if imagename == name:
                    print(imagename, "already in clusterfile")
                    skipflag = True
                    break

            # Skip files that are not .png, .jpeg, nor .jpg
            if imagename.find(".png") == -1 and \
                            imagename.find(".jpeg") == -1 and \
                            imagename.find(".jpg") == -1 or skipflag:
                continue

            print(imagename)
            img = Image.open(original_images_path + "/" + imagename)
            img = img.resize((150, 150))  # optional, to reduce time
            centroids, labels = cluster_image(img,meancount)
            if len(centroids) > 0:
                # Writing to file for verification or tests.
                if isweighted:
                    weightedresult = []
                    weights = bincount(labels)
                    total = sum(weights)
                    i = 0
                    for lab in centroids:
                        # Jacob commented here
                        # weightedresult.append([lab[0], lab[1], lab[2], int(weights[i] * 100 / total)])
                        weightedresult.append([lab[0], lab[1], lab[2]])
                        i = i + 1
                    cluster_list.append([imagename, weightedresult])
                else:
                    cluster_list.append([imagename, result])
                    # clusterfile.write(str([imagename, result]) + "\n")
        except:
            print("Problem processing ", imagename)
    print('Num images in serialized cluster file after processing', len(cluster_list))
    print('Num images processed', count_images)
    if count_images != len(cluster_list):
        print("ERROR! Not all images processed")

    return cluster_list


def add_image_to_clusters(original_images_path, image_path, imagename):
    """
    Checks a directory. If it is a directory, images from the folder 'slices'
    are opened one at a time. Each image is clustered, and its resulting
    cluster points are added to a list.

    Args:
        textfile_read: A file that is open for reading
        textfile_app: A file that is open for appending to the end
        original_images_path: A path that may be used to acess the images
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    """
    clusterfilename = "backend/clusters.txt"
    cluster_list = []

    clusterfile = open("backend/clusters.txt", "a+")
    lines = clusterfile.readlines()

    training_clusters = deserialize()
    exists_in_clusters = 0
    for filename, colors in training_clusters:
        if imagename in filename or imagename == filename:
            print(imagename, "exists in training_clusters")
            exists_in_clusters = 1
            break

    if not exists_in_clusters:
        print(imagename, "does not exist in training_clusters: adding")
        cluster_list = training_clusters

        img = Image.open(image_path + imagename)
        img = img.resize((150, 150))  # optional, to reduce time
        result = cluster_image(img)

        if len(result) > 0:
            # Writing to file for verification or tests.
            cluster_list.append([imagename, result])
            clusterfile.write(str([imagename, result]) + "\n")
        else:
            print("error clustering")
        print(cluster_list)
        serialize(cluster_list)


_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x + y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'


def rgb(triplet):
    triplet = triplet.replace('#', '')
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]


def triplet(rgb, lettercase=LOWERCASE):
    return format(rgb[0] << 16 | rgb[1] << 8 | rgb[2], '06' + lettercase)


def hex_to_rgb(hex):
    return rgb(hex)


from PIL import Image, ImageDraw


def render_palette(palette, filename, mode):
    """ render horizonltally, fix image size,
    vary size of color buckets to fit image size"""
    if len(palette) > 0:
        print(filename, mode)
        # palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
        size = [500, 100]
        bucket_size = 500 / len(palette)
        # print('size', size, len(palette), bucket_size)
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        """[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
        outline – Color to use for the outlin. fill – Color to use for the fill."""
        x = 0
        y = 100  # height
        for k in palette:
            rgb = ()
            if mode == 'rgb':
                rgb = k
            if mode == 'hex':
                if '#' in k:
                    rgb = hex_to_rgb(k[1:])
                else:
                    rgb = hex_to_rgb(k)
            if mode == 'LAB':
                _rgb = ()
                if (type(k) == LabColor):
                    _rgb = convert_color(k, AdobeRGBColor)
                else:
                    # try:

                    l = LabColor(k[0], k[1], k[2])
                    # print('here', l)
                    _rgb = convert_color(l, AdobeRGBColor)
                    # except:
                    #     print('Error converting lab to RGB:', k)

                rgb = _rgb.get_upscaled_value_tuple()
                # print('Converted LAB to RBG:', rgb)
                # rgb = AdobeRGBColor(_rgb[0], _rgb[1], _rgb[2], True)
                # print(rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            if mode == 'HSL':
                _rgb = ()
                if (type(k) == HSLColor):
                    _rgb = convert_color(k, AdobeRGBColor)
                else:
                    print('HSL: bad format')
                    break
                rgb = _rgb.get_upscaled_value_tuple()
                # print('Converted HSL to RBG:', rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            draw.rectangle([(x, 0), (x + bucket_size, y)], fill=rgb)  # , outline=(255,255,255))
            # draw.rectangle([(0,y), (200, y+100)], fill=rgb , outline=(255,255,255))
            x = x + bucket_size
        del draw
        im.save(filename + ".png", "PNG")
    else:
        print('0 length palette!')


def test_cluster_image(image, k_clusters=10):
    img = Image.open(image)
    img = img.resize((150, 150))  # optional, to reduce time
    arr = scipy.misc.fromimage(img)
    shape = arr.shape
    # print(shape, len(shape))
    if len(shape) > 2:
        arr = arr.reshape(scipy.product(shape[:2]), shape[2])
        rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in arr]
        lablist = [convert_color(x, LabColor) for x in rgblist]
        lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
        # print('len(lablist)',len(lablist))
        # print('finding clusters')
        # for i in range(0,5): #try 5 times
        #     #http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.cluster.vq.kmeans.html
        #     codes, dist = scipy.cluster.vq.kmeans(lablist, k_clusters)
        #     print('LEN clusters', len(codes))
        #     #print('codes', codes)
        #     #print('dist', dist)
        #     render_palette(codes, 'DS_clusters_slice'+str(i), 'LAB')
        #     #break
        print('Trying Sci kit')
        # http://scipy-user.10969.n7.nabble.com/kmeans2-question-issue-td1883.html
        for i in range(0, 5):  # try 5 times
            k_means_ex = KMeans(n_clusters=10, n_init=20)
            x = k_means_ex.fit_predict(lablist)
            codes = k_means_ex.cluster_centers_
            # print(x)
            print(len(codes))  # , codes)
            render_palette(codes, 'DS_clusters_SCILEARN_slice' + str(i), 'LAB')
    else:
        print('invalid image')


def cluster_image(image, k_clusters=5):
    '''
    Clusters an image into k cluster points. Then, converts each
    color point from RGB to LAB color format.

    Args:
        image: An open PIL image
        k_clusters: Number of clusters to produce. Defaulted to 10.
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    Replaced vq with sklearn
    '''
    arr = scipy.misc.fromimage(image)
    shape = arr.shape
    if len(shape) > 2:
        arr = arr.reshape(scipy.product(shape[:2]), shape[2])
        rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in arr]
        lablist = [convert_color(x, LabColor) for x in rgblist]
        lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])

        # codes, dist = scipy.cluster.vq.kmeans2(lablist, k_clusters, iter=20)
        # print('LEN clusters', len(codes))
        return cluster_array(k_clusters, lablist)
    else:
        print('Invalid image')
        return [], []


def cluster_array(clusters, lablist):
    #k_means_ex = KMeans(n_clusters=k_clusters)
    k_means_ex = MiniBatchKMeans(n_clusters = clusters)
    # k_means_ex = KMeans(n_clusters=k_clusters)
    x = k_means_ex.fit_predict(lablist)
    codes = k_means_ex.cluster_centers_
    labels = k_means_ex.labels_
    return codes, labels


def cluster_image_predict(image, labpalette, k_clusters=10):
    '''
    Clusters an image into k cluster points. Then, converts each
    color point from RGB to LAB color format.
    Predicts cluster for each color in passed palette
    Palette should be in LAB

    Args:
        image: An open PIL image
        k_clusters: Number of clusters to produce. Defaulted to 10.
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    Replaced vq with sklearn
    '''
    arr = scipy.misc.fromimage(image)
    shape = arr.shape
    if len(shape) > 2:
        arr = arr.reshape(scipy.product(shape[:2]), shape[2])
        rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in arr]
        lablist = [convert_color(x, LabColor) for x in rgblist]
        lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
        # codes, dist = scipy.cluster.vq.kmeans2(lablist, k_clusters, iter=20)
        # print('LEN clusters', len(codes))
        # del dist
        k_means_ex = KMeans(n_clusters=k_clusters)
        x = k_means_ex.fit_predict(lablist)
        codes = k_means_ex.cluster_centers_
        labels = k_means_ex.labels_

        # predict
        predict_labels = k_means_ex.predict(labpalette)
        return codes, labels, predict_labels
    else:
        print('Invalid image')
        return [], [], []


def cluster_multiple_images(images, k_clusters=10):
    '''
    Clusters an image into k cluster points. Then, converts each
    color point from RGB to LAB color format.

    Args:
        images: An array of open PIL image
        k_clusters: Number of clusters to produce. Defaulted to 10.
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    Replaced vq with sklearn
    '''
    fullarr = scipy.misc.fromimage(images[0])
    shape = fullarr.shape
    fullarr = fullarr.reshape(scipy.product(shape[:2]), shape[2])
    # print('fullarr', fullarr.shape)
    for im in images[1:]:
        arr = scipy.misc.fromimage(im)
        shape = arr.shape
        if len(shape) > 2:
            arr = arr.reshape(scipy.product(shape[:2]), shape[2])
            # print(arr.shape)
            fullarr = numpy.concatenate((fullarr, arr), axis=0)
            # print('fullarr',fullarr.shape)
        else:
            print('Invalid image')
    rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in fullarr]
    lablist = [convert_color(x, LabColor) for x in rgblist]
    lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
    k_means_ex = KMeans(n_clusters=k_clusters)
    x = k_means_ex.fit_predict(lablist)
    codes = k_means_ex.cluster_centers_
    labels = k_means_ex.labels_
    return codes, labels


def deserialize(clusterfilename):
    '''
    Uses pickle package to deserialize bytes into a python list object.

    Returns:
        Whatever was containted as bytes, into a python object.
    '''
    # if exists
    if os.path.isfile(clusterfilename):
        cluster_file = open(clusterfilename, "rb")
        try:
            return pickle.load(cluster_file)
        except:
            print('Error reading file or file is empty!')
            return []
    else:
        return []


def serialize(cluster_list, clusterfilename):
    '''
    Uses pickle to convert a parameter into bytes. Writes those bytes to
    the file: clusterlistbytes.

    Args:
        cluster_list: A list object
    '''
    cluster_bytes = open(clusterfilename, "wb")
    # pickle.dump(cluster_list, cluster_bytes)
    pickle.dump(cluster_list, cluster_bytes, protocol=2)


def write_as_text(cluster_list, clusterfilename):
    '''
    to write a readabe file

    Args:
        cluster_list: A list object
    '''
    text_file = open(clusterfilename, "w")
    for item in cluster_list:
        text_file.write("%s\n" % item)
        break
    text_file.close()


import re


def plot_clusters(original_images_path):
    palette_folder = "backend/clusters/"
    color_points = open("backend/cropbox.txt").read()
    training_clusters = deserialize()
    for imagename in os.listdir(original_images_path):
        print(imagename)
        # Obtain start and end points of RGB; save it to string
        start_index = color_points.find(imagename)
        start_index = color_points.find('\n', start_index) + 1
        end_index = color_points.find('\n', start_index)
        rgb_color_str = color_points[start_index:end_index]
        # print(rgb_color_str)
        pal_color_list = []

        # Convert string into list of RGB tuples
        rgb_patt = re.compile("[(](?P<r>.*?)[,][ ](?P<g>.*?)[,][ ](?P<b>.*?)[)]")
        # print(rgb_color_str)
        for res in rgb_patt.finditer(rgb_color_str):
            try:
                pal_color_list.append((int(res.group("r")),
                                       int(res.group("g")),
                                       int(res.group("b"))))
            except:
                print(res.group("b"))
        render_palette(pal_color_list, palette_folder + imagename + "_DS", 'rgb')
        # get clusters
        for filename, colors in training_clusters:
            # print(filename)
            if imagename == filename:
                render_palette(colors, palette_folder + imagename + "_KMEANS", 'LAB')
                break
                # print(current_image_clusters)


def cluster_image_set(folder, k):
    """find and plot clusters - for experiments"""
    # cur_dir = os.path.dirname(os.path.realpath(__file__))
    # print(cur_dir+"/"+folder)
    # print(os.path.isdir(cur_dir+"/"+folder))
    for imagename in os.listdir(folder):
        # try:
        # # Skip files that are not .png, .jpeg, nor .jpg
        # also skip rendered palettes
        if imagename.find("_KMEANS_") > 0 or \
                        imagename.find("_DS_") > 0 or \
                (imagename.find(".png") == -1 and \
                             imagename.find(".jpeg") == -1 and \
                             imagename.find(".jpg") == -1):
            continue

        img = Image.open(folder + "/" + imagename)
        img = img.resize((150, 150))  # optional, to reduce time
        result = cluster_image(img, k_clusters=k)
        # print(result[0])
        palette = [(r, g, b) for r, g, b in result[0]]
        # print(palette)
        render_palette(palette, folder + imagename + "_KMEANS_" + str(k), 'LAB')
        # except:
        #     print("Smth went wrong", imagename)


if __name__ == '__main__':
    # clusterfilename= "backend/clusterlistbytes.txt"
    clusterfilename = "backend/clusterlistbytesweighted.txt"
    training_clusters = deserialize(clusterfilename)
    print('Num images in serialized cluster file', len(training_clusters))
    # # #print(len(training_clusters))
    # overwrite =1
    # clusterfilename_txt= "backend/clusterlistbytesweighted_readable.txt"
    # PATH = r"backend/DESIGNSEEDS/slices"
    # CLUSTERED_LIST = find_clusters_in_dir(training_clusters, PATH, overwrite)
    # serialize(CLUSTERED_LIST, clusterfilename)
    # write_as_text(CLUSTERED_LIST, clusterfilename_txt)
    # print("Processing complete...")
    # palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
    # render_palette(palette, "/home/linka/python/autoimage_flask/backend/original-images/test", 'hex')
    # cluster_image_set("/home/linka/python/autoimage_flask/testing/surveys/6/", 6)
    # plot_clusters("backend/DESIGNSEEDS/slices/")
    # test_cluster_image("testing/bw.png", 10)
    # test_cluster_image("backend/DESIGNSEEDS/PineappleHues.png", 10)
    # add_image_to_clusters( r"backend/DESIGNSEEDS/slices", "/home/linka/python/autoimage_flask/testing/selected_images/", "macke1.png")