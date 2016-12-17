from PIL import Image
from glob import glob
import re
from scipy import misc
import numpy as np
from collections import defaultdict
import pallette_manager
from skimage import color
from random import randint


palettesdir = '/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/palettes/'
imagesdir = '/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/slices_resized/'
palettefile = 'file_palette_RGBmapping.txt'

palettesdict = pallette_manager.readRGBValuesFromFile(filename = 'file_palette_RGBmapping.txt')
filelist = list(palettesdict.keys())

def ExtractSingleImageRGBIntoNumphy(filepath = '/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/slices_resized/AboardColor610.png'):
    #img = Image.open(filepath)
    imgarray = misc.imread(filepath, mode='RGB')
    flat_imgarray = imgarray.flatten()
    #print(imgarray)
    #print(imgarray.shape)
    #print(imgarray.size)
    #print(len(imgarray))
    #print(len(imgarray[0]))
    #print(len(imgarray[0][0]))
    #print(flat_imgarray.shape)
    return flat_imgarray

def ExtractSingleImageLABIntoNumphy(filepath = '/media/jacob/Elements/MyStuff/Chama/DESIGNSEEDS/DESIGNSEEDS/slices_resized/AboardColor610.png'):
    #img = Image.open(filepath)
    imgarray = misc.imread(filepath, mode='RGB')
    labimagearray = color.rgb2lab(imgarray)
    flat_imgarray = labimagearray.flatten()
    return flat_imgarray


# got this function from https://gist.github.com/perrygeo/4512375#file-normalize_numpy-py-L10
# it can be used for linear scaling between 2 ranges
def scale_linear(rawpoints, fromhigh=255.0, fromlow=0.0,tohigh=1.0,tolow=0.0):
    rng = fromhigh - fromlow
    return tohigh - (((tohigh - tolow) * (fromhigh - rawpoints)) / rng)

def ExtractPalleteColoursToNumpy(filename,palcount):
    paletteslist = palettesdict[filename]
    if palcount > len(paletteslist):
        raise ValueError('Number of palettes found less than requested.')
    pal_array = np.asarray(paletteslist[0:palcount],dtype = float)
    flatpal_array = pal_array.flatten()

    #normalize array before returning
    return scale_linear(flatpal_array)

def ExtractPalleteColoursCountToNumpy(filename,classcount):
    paletteslist = palettesdict[filename]
    palcount = len(paletteslist)
    if 2 > palcount:
        raise ValueError('Number of palettes found less than 2.')
    pal_array = np.asarray(paletteslist[0:palcount],dtype = float)
    flatpal_array = pal_array.flatten()
    #create one hot vector of the count
    return dense_to_one_hot(palcount,classcount)


def dense_to_one_hot(count, num_classes):
  """Convert class labels from scalars to one-hot vectors."""

  #index_offset = numpy.arange(num_labels) * num_classes
  # i have reduce the number of classes from 5-10 but as no
  # palettes exist with less than 5 count
  count = count -5
  labels_one_hot = np.zeros(num_classes)
  labels_one_hot.flat[count] = 1
  return labels_one_hot

BATCHDONECOUNT = 0
# the file list has all the names of the palette files and so is traversed through is a batch wise fashion based on
# the batchcount
def get_batch(batchcount):
    global BATCHDONECOUNT
    how_far = BATCHDONECOUNT + batchcount if BATCHDONECOUNT + batchcount < len(palettesdict) else len(palettesdict)
    processed_count = 0
    PaletteCount = 5
    for i in range(BATCHDONECOUNT,how_far):
        if processed_count == 0:
            x_array = ExtractSingleImageRGBIntoNumphy(imagesdir + filelist[i])
            y_array = ExtractPalleteColoursToNumpy(filelist[i],PaletteCount)
        else:
            #x_array = np.append(x_array,ExtractSingleImageRGBIntoNumphy(imagesdir + filelist[i]),axis=0)
            x_array = np.column_stack([x_array, ExtractSingleImageRGBIntoNumphy(imagesdir + filelist[i])])
            y_array = np.column_stack([y_array, ExtractPalleteColoursToNumpy(filelist[i],PaletteCount)])
        palettesdict[filelist[i]]
        processed_count += 1
    returnbatch = []
    #print(x_array)

    #datacount,imgcount = x_array.shape
    returnbatch.append(x_array.transpose())

    #print(x_array.shape)
    returnbatch.append(y_array.transpose())
    BATCHDONECOUNT = how_far
    if (how_far == len(palettesdict)):
        BATCHDONECOUNT = 0
    return returnbatch

# This function reads images and their corresponding palette in a sequential manner. If this reads all the
# available images it just resets and reads files from the beginning. Note that this may give less number of images
# in the last step before it resets to 0 count
def get_countbatchrotation(batchcount):
    global BATCHDONECOUNT
    how_far = BATCHDONECOUNT + batchcount if BATCHDONECOUNT + batchcount < len(palettesdict) else len(palettesdict)
    processed_count = 0
    classCount = 5
    for i in range(BATCHDONECOUNT,how_far):
        if processed_count == 0:
            x_array = ExtractSingleImageLABIntoNumphy(imagesdir + filelist[i])
            y_array = ExtractPalleteColoursCountToNumpy(filelist[i],classCount)
        else:
            #x_array = np.append(x_array,ExtractSingleImageLABIntoNumphy(imagesdir + filelist[i]),axis=0)
            x_array = np.column_stack([x_array, ExtractSingleImageLABIntoNumphy(imagesdir + filelist[i])])
            y_array = np.column_stack([y_array, ExtractPalleteColoursCountToNumpy(filelist[i],classCount)])
        palettesdict[filelist[i]]
        processed_count += 1
    returnbatch = []
    #print(x_array)

    #datacount,imgcount = x_array.shape
    returnbatch.append(x_array.transpose())

    #print(x_array.shape)
    returnbatch.append(y_array.transpose())
    BATCHDONECOUNT = how_far
    if (how_far == len(palettesdict)):
        BATCHDONECOUNT = 0
    return returnbatch

def get_countbatch(batchcount):
    if (batchcount > len(filelist)):
        raise ValueError('Not enough images in data to show')

    processed_count = 0
    classCount = 5
    chosenfilelist = []

    for _ in range(batchcount):
        chosenimage  = filelist[randint(0,len(filelist)-1)]
        while chosenimage in chosenfilelist:
            chosenimage = filelist[randint(0,len(filelist)-1)]
        chosenfilelist.append(chosenimage)

    for chosenimage in chosenfilelist:
        if processed_count == 0:
            x_array = ExtractSingleImageLABIntoNumphy(imagesdir + chosenimage)
            y_array = ExtractPalleteColoursCountToNumpy(chosenimage,classCount)
        else:
            x_array = np.column_stack([x_array, ExtractSingleImageLABIntoNumphy(imagesdir + chosenimage)])
            y_array = np.column_stack([y_array, ExtractPalleteColoursCountToNumpy(chosenimage,classCount)])
        processed_count += 1
    returnbatch = []
    #print(x_array)
    #datacount,imgcount = x_array.shape
    returnbatch.append(x_array.transpose())
    #print(x_array.shape)
    returnbatch.append(y_array.transpose())
    return returnbatch


if __name__ == '__main__':
    for _ in range(100):
        batch = get_batch(100)
        print(batch)
        print('batchdonecount' + str(BATCHDONECOUNT))

'''
print('batchdonecount' + str(BATCHDONECOUNT))
batch = get_batch(50)
print(batch)
print('batchdonecount' + str(BATCHDONECOUNT))
batch2 = get_batch(50)
print(batch2)
print('batchdonecount' + str(BATCHDONECOUNT))
'''

# Examples on how to run linear normalization on output data
#print(batch[0])
#print(batch[1])
#print(scale_linear(batch[1]))
#print(scale_linear(scale_linear_bycolumn(batch[1]),1.0,0.0,255.0,0.0))







