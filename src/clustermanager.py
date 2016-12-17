"""
This module is used to create and access the image color clustering
information for images in the processingfolderpath
"""
import os
import os.path
import pickle
import random
import scipy
import numpy as np
from numpy import bincount
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
import pallette_manager as pm
from configmanager import Configs

# Just a list holding the names of the clustering used
clustering_names = ['KMeans', 'MiniBatchKMeans', 'Random']

# path where all the data is
# this folder should have a folder called /slices where the images are located
processing_folder_path = Configs.ProcessingFolderPath

# Using convention for means files:
# kmeansfilepath = processing_folder_path + kmeansfileprefix + '_' + meancount
# e.g. /home/jacob/PycharmProjects/Chameleon/images/means_KMeans
meansfileprefix = r'means'


def find_clusters_in_dir(training_clusters, original_images_path,
                         calgorithm='KMeans'):
    """
    Checks a directory. If it is a directory, images from the folder 'slices'
    are opened one at a time. Each image is clustered, and its resulting
    cluster points are added to a list.

    Args:
        training_clusters:
        original_images_path: A path that may be used to acess the images
        overwrite: unused flag
        calgorithm:
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
        skipflag = False
        count_images = count_images + 1

        # Skip files that are not .png, .jpeg, nor .jpg
        if imagename.find(".png") == -1 and \
           imagename.find(".jpeg") == -1 and \
           imagename.find(".jpg") == -1 or skipflag:
            continue

        print(imagename)
        img = Image.open(original_images_path + "/" + imagename)
        img = img.resize((150, 150))  # optional, to reduce time

        # Jacob: get the artist palette count here assume 5 for now
        mean_count = pm.getArtistPaletteCount(imagename)
        if mean_count == 0:
            # hard code means to 10 in case image doesn't exist(prevents abort)
            mean_count = 10
        centroids, labels = cluster_image(img, mean_count, calgorithm)
        if len(centroids) > 0:
            # Writing to file for verification or tests.
            if isweighted:
                weightedresult = []
                weights = bincount(labels)
                sum(weights)
                i = 0
                for lab in centroids:
                    # Jacob commented here
                    # weightedresult.append([lab[0], lab[1], lab[2],
                    #                       int(weights[i] * 100 / total)])
                    weightedresult.append([lab[0], lab[1], lab[2]])
                    i = i + 1
                cluster_list.append([imagename, weightedresult])
            else:
                result = None
                cluster_list.append([imagename, result])
                # clusterfile.write(str([imagename, result]) + "\n")
        # except:
        #    print("Problem processing ", imagename)
    print('Num images in serialized cluster file after processing',
          len(cluster_list))
    print('Num images processed', count_images)
    if count_images != len(cluster_list):
        print("ERROR! Not all images processed")

    return cluster_list


def cluster_image(image, k_clusters=5, calgorithm='KMeans'):
    """
    Clusters an image into k cluster points. Then, converts each
    color point from RGB to LAB color format.

    Args:
        image: An open PIL image
        k_clusters: Number of clusters to produce. Defaulted to 10.
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    Replaced vq with sklearn
    """
    arr = scipy.misc.fromimage(image)
    shape = arr.shape
    if len(shape) > 2:
        arr = arr.reshape(scipy.product(shape[:2]), shape[2])
        rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in arr]
        lablist = [convert_color(x, LabColor) for x in rgblist]
        lablist = np.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
        return cluster_array(k_clusters, lablist, calgorithm)


def cluster_array(clusters, lablist, calgorithm):
    """

    """
    if calgorithm == 'KMeans':
        means_ex = KMeans(n_clusters=clusters)
        means_ex.fit_predict(lablist)
        codes = means_ex.cluster_centers_
        labels = means_ex.labels_
        return codes, labels
    elif calgorithm == 'MiniBatchKMeans':
        means_ex = MiniBatchKMeans(n_clusters=clusters)
        means_ex.fit_predict(lablist)
        codes = means_ex.cluster_centers_
        labels = means_ex.labels_
        return codes, labels
    elif calgorithm == 'Random':
        means = []
        labels = []
        for i in range(clusters):
            means.append(random.choice(lablist))
            labels.append(i)
        return means, labels


def deserialize(clusterfilename):
    """
    Uses pickle package to deserialize bytes into a python list object.

    Returns:
        Whatever was containted as bytes, into a python object.
    """
    if os.path.isfile(clusterfilename):
        cluster_file = open(clusterfilename, "rb")
        try:
            return pickle.load(cluster_file)
        except Exception:
            print('Error reading file or file is empty!')
            return []
    else:
        return []


def serialize(cluster_list, clusterfilename):
    """
    Uses pickle to convert a parameter into bytes. Writes those bytes to
    the file: clusterlistbytes.

    Args:
        cluster_list: A list object
    """
    cluster_bytes = open(clusterfilename, "wb")
    # pickle.dump(cluster_list, cluster_bytes)
    pickle.dump(cluster_list, cluster_bytes, protocol=2)


def findimagekmeans(imagename):
    """

    :param imagename:
    :return: means for the kmeans clustering of that image
    """
    return findmeansfromlist(imagename, k_means_data)


def findimageminibatchmeans(imagename):
    """

    :param imagename:
    :return: means for the minibatch-kmeans clustering of that image
    """
    return findmeansfromlist(imagename, mini_batch_data)


def findimagerandommeans(imagename):
    """
    """
    return findmeansfromlist(imagename, random_data)


def write_Clustering_Information_to_file(kmeanslist, processingfolderpath,
                                         calgorithm):
    """
    This function writes the means information in a temporary file in
    processing_folder_path
    :param kmeanslist:
    :param processingfolderpath:
    :param overwrite:
    :param calgorithm:
    :return:
    """
    imagesfolderpath = processingfolderpath + r'slices/'
    find_clusters_in_dir(kmeanslist, imagesfolderpath, calgorithm)
    serialize(kmeanslist, processingfolderpath + meansfileprefix + '_' +
              calgorithm)


# checks if file kmeans file exists
# reads all the values for k means and returns
def read_Clustering_information_from_file(calgorithm):
    """
    This function reads all the information about a specific mean type for all
    images from file and returns list
    :param calgorithm:
    :return: list of means information for images
    """
    kmeanslist = []
    kmeansfile = processing_folder_path + meansfileprefix + '_' + calgorithm
    if not os.path.exists(kmeansfile):
        write_Clustering_Information_to_file(kmeanslist,
                                             processing_folder_path,
                                             calgorithm)
    return deserialize(kmeansfile)

k_means_data = read_Clustering_information_from_file('KMeans')
mini_batch_data = read_Clustering_information_from_file('MiniBatchKMeans')
random_data = read_Clustering_information_from_file('Random')
# birchdata = readClusteringinformationfromfile('Birch')
# warddata = readClusteringinformationfromfile('Ward')


def findmeansfromlist(imagename, meanslist):
    """
    This function returns the mean information for image from meanslist
    :param imagename:
    :param meanslist:
    :return:
    """
    for item in meanslist:
        if item[0] == imagename:
            return np.array(item[1])
