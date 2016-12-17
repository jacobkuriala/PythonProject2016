import pickle
import clustermanager
import os
import pallette_manager as pm
from Chameleon import matching
from configmanager import Configs


# processing_folder_path = r'/home/jacob/PycharmProjects/Chameleon/images/'
scorefilename = 'palettescore'
processingfolderpath = Configs.ProcessingFolderPath

def readscoresfromfile():
    """
    This function reads the palette information from palettesscore file
    :return:
    """
    if not os.path.isfile(processingfolderpath+scorefilename):
        writescorestofile()
    return deserialize(processingfolderpath+scorefilename)

def writescorestofile():
    """
    This funciton writes the palette information to the palettesscore file
    :return:
    """
    palettescores = {}
    palettes_dict = pm.read_palette_colors_file()

    for artistpalettekey, artistpalettevalue in palettes_dict.items():
        kmeanspalette = clustermanager.findimagekmeans(artistpalettekey)
        minibatchmeanspalette = clustermanager.findimageminibatchmeans(artistpalettekey)
        randommeanspalette = clustermanager.findimagerandommeans(artistpalettekey)

        kmeansscore = calculateDistance(artistpalettevalue, kmeanspalette)
        minibatchmeansscores = calculateDistance(artistpalettevalue, minibatchmeanspalette)
        randommeansscore = calculateDistance(artistpalettevalue, randommeanspalette)

        palettescores[artistpalettekey + '_kmeans'] = kmeansscore
        palettescores[artistpalettekey + '_minibatchmeans'] = minibatchmeansscores
        palettescores[artistpalettekey + '_randmeans'] = randommeansscore
        palettescores[artistpalettekey + '_designer'] = 0.0
    normalizevaluesandgeneratescores(palettescores)
    serialize(palettescores, processingfolderpath+scorefilename)
    #print(palettescores)

def deserialize(clusterfilename):
    '''
    Uses pickle package to deserialize bytes into a python list object.

    Returns:
        Whatever was containted as bytes, into a python object.
    '''
    if os.path.isfile(clusterfilename):
        cluster_file = open(clusterfilename, "rb")
        try:
            return pickle.load(cluster_file)
        except:
            print('Error reading file or file is empty!')
            return []
    else:
        return []

def serialize(obj,filename):
    '''
    Uses pickle to convert a parameter into bytes. Writes those bytes to
    the file: clusterlistbytes.

    Args:
        obj: A list object
    '''
    cluster_bytes = open(filename, "wb")
    pickle.dump(obj, cluster_bytes, protocol=2)

def calculateDistance(palette1,palette2):
    """
    Calculates the disctance betweeen 2 palettes
    This function internally calls the distance calcualtion
    available in chameleon
    :param palette1:
    :param palette2:
    :return:
    """
    return matching.match_and_calculate_distance(palette1, palette2)

def normalizevaluesandgeneratescores(mydict):
    """
    Normalizes the score based on the max and min range of
    the scores
    :param mydict:
    :return:
    """
    maxval = max(mydict.values())
    minval = min(mydict.values())
    for key,value in mydict.items():

        normaldist = (value-minval)/(maxval - minval)
        mydict[key] = 1 - normaldist


scoresdict = readscoresfromfile()


def getscore(imagename_meansmethod):
    """
    This function can directly retaurn the
    mean score of imagename_meanmethod
    (The readscoresfromfile reads this information
    and stores in the dictionary
    :param imagename_meansmethod:
    :return:
    """
    return scoresdict[imagename_meansmethod]
