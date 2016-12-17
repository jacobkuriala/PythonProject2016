'''
Created on Nov 28, 2015

@author: pvolkova

'''
from PIL import Image
from PIL import ImageDraw
import os
import sqlite3 as lite
import numpy as np

#from nearest_palette import create_distance_matrix

import csv

def get_websites(savefolder, weblist ):
    con =0#connect_to_db()
    for rank, name, website in weblist:
        #try:
        print(rank, name, website)
        #this downloads images folder
        try:
            download_website(savefolder, website)
        except:
            print(website, 'problem')
        #this downloads images referenced in html
        try:
            download_website_images(os.path.join(savefolder, 'images'), website, con)
        except:
             print('Error getting website', rank, name, website)
        # break
    if con:
        con.close()

def get_website_training_set():
    """save all fortune 500 websites"""
    savefolder = 'backend/trainingset/Fortune500/'
    with open('backend/Fortune500.csv', 'r') as f:
        reader = csv.reader(f)
        fortune500 = list(reader)
        #fortune500=[[0, 'test', 'www.tractorsupply.com'], [0, 'test', 'www.tractorsupply.com']]
        #print(your_list)
        get_websites(savefolder, fortune500[1:])

def get_website_test_set():
    """save all fortune 500 websites"""
    savefolder = 'backend/testset'
    with open('backend/Fortune500.csv', 'r') as f:
        reader = csv.reader(f)
        fortune500 = list(reader)
        get_websites(savefolder, fortune500)#[76:])


def download_website(savefolder, website):
    command = 'wget --no-directories --no-clobber --no-parent --random-wait --page-requisites --user-agent="Mozilla/5.0" -P -E -e robots=off'+savefolder+'  '+website
    os.system(command)

import re
#from html.parser import HTMLParser  
from urllib.request import Request, urlopen  
#from urllib import parse
def download_website_images(savefolder, website, con):
    req = Request('http://'+website, headers={'User-Agent': 'Mozilla/5.0'})
    os.environ['http_proxy']=''
    htmlString=urlopen(req).read().decode('utf-8')
    #print(htmlString)
    patt = re.compile(r"""(?P<imageurl>(http://){1}.+?((\.jpg)|(\.jpeg)|(\.gif)|(\.png)|(\.bmp)){1})""", re.IGNORECASE)
    for res in patt.finditer(htmlString):
        imageurl= res.group("imageurl")
        #print( imageurl)
        os.system('wget -P '+savefolder+' '+imageurl)
        #recordede in get_web_images_training_set
        #record_image(con, savefolder, website, imageurl)

def allowed_file(fileusername):
    extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    # if '.' in fileusername: 
    #     return (fileusername.rsplit('.', 1)[1]).lower() in extensions
    result = False
    for e in extensions:
        #print(e)
        if e in fileusername:
            result= True
            break
    #print(result)
    return result

def get_web_images_set(tmpfolder, savefolder):
    """ 
    Get all images from all saved websites in folder
    Rename duplicates
    Save them to a separate folder and record in the database
    """
    # tmpfolder = 'backend/trainingset/Fortune500/www.chevron.com/'
    # savefolder = 'backend/trainingset/Fortune500/images'
    con =connect_to_db()
    for root, dirs, files in os.walk(tmpfolder):
        for f in files:
            #print(root,f)
            try:
                if allowed_file(f):
                    # try:
                    print(f, 'is allowed')
                    with Image.open(os.path.join(root,f)) as im:
                        #print(f, im.format, "%dx%d" % im.size, im.mode)
                        #save to folder
                        im.save(os.path.join(savefolder, f))
                        #record to database
                        error =record_image(con, savefolder, root, f)
                        print(error)
                    # except:# IOError:
                    #     #print('error')
                    #     pass
            except:# IOError:
                #print('error')
                pass
    if con:
        con.close()


def record_image(con, path, source, name):
    Error =''
    try:
        command = "INSERT INTO Image VALUES ( \'"+path+"\', \'"+source+"\', \'"+name+"\');"
        con.execute(command)
        con.commit()
    except lite.Error as e:        
        logging.info("Error %s:" % e.args[0])
        Error = str(e)
    return Error

def connect_to_db():
    con = None
    Error =''
    try:
        con = lite.connect('training.db')
        con.execute('''create table if not exists Image (id integer primary key autoincrement, path TEXT, source TEXT, name TEXT)''')
        con.execute('''create table if not exists ImageFeatures (id integer primary key autoincrement, path TEXT, source TEXT, name TEXT)''')
        con.commit()
        #con.execute('''create table if not exists ImageFeatures (id integer primary key autoincrement, path TEXT, source TEXT, name TEXT)''')
        con.commit()
    except lite.Error as e:        
        logging.info("Error %s:" % e.args[0])
        Error = str(e)
    return con

# def get_images_features(istraining,savefolder, textfile):
#     """
#     Read image from file
#     Get image features and record in the database
#     """
#     # savefolder = 'backend/trainingset/Fortune500/images'
#     # textfile ="trainingset"
#     trainingfile = open(os.path.join(savefolder, textfile) , "r")
#     lines = trainingfile.readlines()
#     image_features =[]
#     image_recolorability =[]
#     image_list =[]
#     for l in lines:
#         if istraining:
#             flag, name = l.split(' ')
#             print(flag, name)
#             name= name.strip('\n')
#             if flag=='x':
#                 print(flag, ', skipping')
#                 continue
#             if flag=='y':
#                 image_recolorability.append(1)
#             if flag=='n':
#                 image_recolorability.append(0)
#             image_list.append(name)
#         else:
#             name = l.strip('\n')
#             image_list.append(name)
#         #get image features
#         features= get_image_features(savefolder, name)
#         image_features.append(features)
#         #break
#     trainingfile.close()
#     #write features to database
#     write_features(image_features,  image_recolorability,  image_list, savefolder)

def get_all_image_features(istraining,savefolder):
    """
    Read image from folder
    Get image features and record 
    """
    image_features =[]
    image_recolorability =[]
    image_list =[]
    for root, dirs, files in os.walk(savefolder):
        for f in files:
            try:
                if allowed_file(f):
                    if istraining:
                        if os.path.basename(savefolder)=='recolorable':
                            image_recolorability.append(1)
                        if os.path.basename(savefolder)=='nonrecolorable':
                            image_recolorability.append(0)
                    image_list.append(f)
                    #get image features
                    features= get_image_features(savefolder, f)
                    if len(features)>0:
                        image_features.append(features)
            except:
                print('Problem with', f)
    return image_features,  image_recolorability,  image_list, savefolder

def get_trainingset_features(mainfolder):    
    #write features 
    """calculates and writes features """
    image_features =[]
    image_recolorability =[]
    image_list =[]
    image_features,  image_recolorability,  image_list, folder=get_all_image_features(1,  os.path.join(mainfolder, 'recolorable'))
    image_features1,  image_recolorability1,  image_list1, folder=get_all_image_features(1 , os.path.join(mainfolder, 'nonrecolorable'))
    write_features(image_features+image_features1,  image_recolorability+image_recolorability1,  image_list+image_list1, mainfolder)

def get_image_features(savefolder, name):
    features =[]
    try:
        image = Image.open(os.path.join(savefolder, name))  
        # for h in image.histogram():  
        #     features.append(h)
        # number of colors
        features.append(get_num_color(image))
        # aspect ratio
        features.append(float(image.size[0]/image.size[1]))
        # size
        features.append(image.size[0])
        features.append(image.size[1])
        try:
            im = cv2.imread(os.path.join(savefolder, name))
        except:
            pass

        if im!=None:
            #check if grayscale
            if len(im.shape)==2:
                features.append(1)
            else:
                features.append(0)
            histogram=get_histogram(im)
            for h in histogram:
                features.append(h)
        else:
            features.append(-1)
            for h in range(0, 64):
                features.append(0)
        # texture
        # open cv face detector
        # print(features)
        # print(len(features))
    except:
        print('bad image file', name)
    return features


import pickle
def write_features(image_features, image_recolorability,  image_list, folder):
    """serizlize python object"""
    f = open(os.path.join(folder, 'features'), 'wb')
    pickle.dump(image_features, f)
    f = open(os.path.join(folder, 'image_list'), 'wb')
    pickle.dump(image_list, f)
    if len(image_recolorability)>0:
        f = open(os.path.join(folder, 'recolorability'), 'wb')
        pickle.dump(image_recolorability, f)      
    f.close()  

def get_features(folder):
    """serizlize python object"""
    f = open(os.path.join(folder, 'features'), 'rb')
    image_features=pickle.load(f)
    f = open(os.path.join(folder, 'image_list'), 'rb')
    image_list=pickle.load(f)
    if os.path.exists(os.path.join(folder, 'recolorability')):
        f = open(os.path.join(folder, 'recolorability'), 'rb')
        image_recolorability=pickle.load(f)
    else:
        image_recolorability =[]
    f.close()     
    #print(image_features[0], image_recolorability[0], image_list[0])
    return image_features, image_recolorability, image_list

def get_main_color(file):
    """get main color in the image """
    img = Image.open(file)
    #Returns a list of colors used in this image.
    colors = img.getcolors(256)
    if colors==None:
        return None
    else:
        max_occurence, most_present = 0, 0
        try:
            for c in colors:
                if c[0] > max_occurence:
                    (max_occurence, most_present) = c
            return most_present
        except TypeError:
            raise Exception("Too many colors in the image")

def get_num_color(img):
    """get number of colors in the image - None is > 256"""
    #Returns a list of colors used in this image.
    colors = img.getcolors(256)
    if colors==None:
        return 1000
    else:
        return len(colors)

import numpy as np
import cv2
def get_histogram(image):
    """3D color histogram in the HSV color space (Hue, Saturation, Value)
    4bins for the Hue channel, 4 bins for the saturation channel, and 4 bins for the value channel
    yielding a total feature vector of dimension 4 x 4 x 4 """
    # convert the image to the HSV color space and initialize
    # the features used to quantify the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bins=[4, 4, 4]
    # extract a 3D color histogram from the masked region of the
    # image, using the supplied number of bins per channel; then
    # normalize the histogram - relative percentage counts for a particular bin and not the integer counts
    #cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]]) 
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    #print('initial hist',hist.shape, hist)
    hist_norm = np.array(hist.shape)
    hist_norm=cv2.normalize(hist, hist)
    #print('normalized', hist_norm.shape, hist_norm)
    # hist_norm.flatten()
    #flatten
    hist = list(flatten(flatten_np(hist_norm)))
    #print('flat', len(hist), hist)
    return hist

def flatten(container):
    for i in container:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in flatten(i):
                yield j
        else:
            yield i

def flatten_np(items, seqtypes=(list, tuple)):
    _items=[]
    for i, x in enumerate(items):
        y = list(x.flatten())
        _items.append(y)
    return _items

def get_image_names(savefolder, textfile):
    """Create list of images for marking
    x means bad or duplicate - exclude
    y menas recolor
    n means do not recolor"""
    # savefolder = 'backend/trainingset/Fortune500/images'
    # textfile ="trainingset"
    filenames =[]
    for root, dirs, files in os.walk(savefolder):
        for f in files:
            filenames.append(f)
    trainingfile = open(os.path.join(savefolder, textfile) , "w")
    for n in sorted(filenames):
        trainingfile.write(n+ '\n')
    trainingfile.close()

##################   classifier ################################
# from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import  RandomForestClassifier
from sklearn import metrics

def predict_recolorability(folder, imagename):
    model= train_classifier()
    result =0
    features = get_image_features(folder, imagename)
    if len(features)>0:
        np_features=np.array(features)
        result = model.predict(np_features) 
        print(folder, imagename, result)
    return result

def predict_recolorability_use_model(folder, imagename, model):
    result =0
    features = get_image_features(folder, imagename)
    if len(features)>0:
        np_features=np.array(features)
        result = model.predict(np_features) 
    return result

import pickle
def train_classifier(rerun_features =0):  
    #read traning set
    trainfolder = 'backend/trainingset/'
    #should not be necessary to recalculate features each time
    if rerun_features==1:
        get_trainingset_features(trainfolder)
    trainingset, trainingset_recolorability, trainingset_image_list = get_features(trainfolder)
    # np_trainingset=np.empty((len(trainingset),len(trainingset[0])), dtype=float)
    # for i in range(0, len(trainingset)-1):
    #     np_trainingset[i,:] = np.array(trainingset[i])
    np_trainingset=np.array(trainingset)
    label = np.array(trainingset_recolorability)
    #classify
    #model =RandomForestClassifier()
    #model = LogisticRegression()
    #model = KNeighborsClassifier()
    #model = GaussianNB()
    model = DecisionTreeClassifier()
    #model = SVC()
    model.fit(np_trainingset, label)
    # summarize the fit of the model
    #print(label)
    #expected = label
    #predicted = model.predict(np_trainingset)
    #print(predicted)
    #print(metrics.classification_report(expected, predicted))
    #print(metrics.confusion_matrix(expected, predicted))
    return model

def write_model(model, destination):
    dmfile = open(destination, 'wb')
    pickle.dump(model,dmfile)
    dmfile.close()

def get_model(path):
    dmfile = open(path, "rb")
    model = pickle.load(dmfile)
    return model

def test_classifier():
    """this is not a real test because it's trained and tested on same data"""
    #read test set
    testfolder = 'backend/testset/images'
    testset, test_recolorability, test_image_list = get_image_features(testfolder)
    np_testset = np.array(testset)
    model= train_classifier()
    result = model.predict(np_testset) 
    print(result)
    #write result

import shutil, random
def test_classifier():
    """Split training set into training and test randomly
    Train classifier on training 
    Test on test
    Report results"""
    #read test set
    folder = '/home/linka/python/autoimage_flask/backend/testing_classifier/'
    trainfolder = os.path.join(folder, 'trainset')#recolorable
    testfolder = os.path.join(folder, 'testset')#recolorable
    sourcefolder = '/home/linka/python/autoimage_flask/backend/trainingset/'
    #Split training set into training and test randomly
    # filenames =[]
    # for root, dirs, files in os.walk(os.path.join(sourcefolder, 'recolorable')):
    #     for f in files:
    #         filenames.append(f)
    # test= random.sample(filenames, int(len(filenames)/2))
    # testsetsize = len(test)
    # print('test size for recolorable', testsetsize)
    # for f in filenames:
    #     if(f in test):
    #         shutil.copy2(os.path.join(sourcefolder, 'recolorable',f), os.path.join(testfolder, 'recolorable',f)) 
    #     else:
    #         shutil.copy2(os.path.join(sourcefolder, 'recolorable',f), os.path.join(trainfolder, 'recolorable',f)) 
    # filenames =[]
    # for root, dirs, files in os.walk(os.path.join(sourcefolder, 'nonrecolorable')):
    #     for f in files:
    #         filenames.append(f)
    # test= random.sample(filenames, int(len(filenames)/2))
    # testsetsize = len(test)
    # print('test size for nonrecolorable', testsetsize)
    # for f in filenames:
    #     if(f in test):
    #         shutil.copy2(os.path.join(sourcefolder, 'nonrecolorable',f), os.path.join(testfolder, 'nonrecolorable',f)) 
    #     else:
    #         shutil.copy2(os.path.join(sourcefolder, 'nonrecolorable',f), os.path.join(trainfolder, 'nonrecolorable',f)) 
    #Train classifier on training 
    #get_trainingset_features(trainfolder) 
    model= train_classifier()
    trset, train_recolorability, tr_image_list = get_features(trainfolder)
    np_trset = np.array(trset)
    result = model.predict(np_trset) 
    label = np.array(train_recolorability)
    print('result on training set')
    print(metrics.classification_report(label, result))
    #get_trainingset_features(testfolder)
    testset, test_recolorability, test_image_list = get_features(testfolder)
    np_testset = np.array(testset)
    result = model.predict(np_testset) 
    label = np.array(test_recolorability)
    print('result on testset')
    print(metrics.classification_report(label, result))

################## end classifier ################################

if __name__ == '__main__':
    #tmpfolder ="/home/linka/python/autoimage_flask/testing/2_recolor/input_web/w5"
    #get_web_images(tmpfolder)
    #et_website_training_set()
    #download_website('backend/trainingset/Fortune500/images/', 'http://www.apple.com/Apple.html')
    #get_web_images_set('backend/testset', 'backend/testset/images')
    # get_image_names('backend/testset/images', 'testset')
    # get_image_names('backend/testset/images1', 'testset')
    #get_images_features(0, 'backend/testset/images', 'testset')    
    #get_image_features('backend/testinput/images', 'Lake_Shore.jpg')
    get_trainingset_features('backend/trainingset/')
    #get_website_test_set()
    # features, image_recolorability, image_list= get_features('backend/trainingset/') #Fortune500/images')
    # convert_feature_dict_to_numpy(features, image_recolorability, image_list)
    model = train_classifier()
    write_model(model, 'img_recolor_model')
    # #test_classifier()
    #predict_recolorability('backend/testset/images', 'young_couple_laughing_230x130.jpg')
    #allowed_file('backend/trainingset/Fortune500/www.chevron.com/images/home/features Feature-imgHomeStory2014CR.jpg')
    #test_classifier()