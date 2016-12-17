
from __future__ import print_function
from featureio import read_features
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
import scorer
import pallette_manager as pm
import featureio as fio
import clustermanager as cm
import matplotlib.image as mpimag
import matplotlib.pyplot as plt
import skimage as ski


def get_data():
    feature_dict = read_features()
    palette_feature_list = []
    palette_score_list = []
    for palette_name, feature_list in feature_dict.items():
        palette_feature_list.append(feature_list)
        palette_score_list.append(scorer.getscore(palette_name))

    return palette_feature_list, palette_score_list


def train_model():
    features, scores = get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, scores,
                                                        test_size=0.4,
                                                        random_state=0)

    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    return lasso_model, X_test, y_test

def test_model():
    model, x_test, y_test = train_model()
    predicted_scores = model.predict(x_test)
    fig, ax = plt.subplots()
    ax.scatter(y_test, predicted_scores)
    ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4)
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Actual', fontsize=18)
    ax.set_ylabel('Predicted',fontsize=18)
    ax.set_title('All features',fontsize=25)
    plt.show()
    print("Coefficient of Determniaiton: " + str(model.score(x_test, y_test)))


def production_model():
    features, scores = get_data()
    Badpalette = np.array([[[0.0, 0.0, 0.0],
                            [163.0, 73.0, 164.0],
                            [205.0, 16.0, 26.0],
                            [103.0, 152.0, 114.0],
                            [63.0, 72.0, 204.0]]], dtype=np.uint8)
    AnotherBadpalette = np.array([[[225.0, 225.0, 0.0],
                                 [0.0, 255.0, 255.0],
                                 [255.0, 0.0, 255.0],
                                 [0.0, 255.0, 64.0],
                                 [255.0, 0.0]]], dtype=np.uint8)
    goodpalette = np.array([[[221.0, 135.0, 210.0],
                             [135.0, 169.0, 221.0],
                             [135.0, 221.0, 156.0],
                             [218.0, 221.0, 135.0],
                             [49.0, 59.0, 79.0]]], dtype=np.uint8)

    lab_bad_palette = ski.color.rgb2lab(Badpalette)[0]
    lab_good_palette = ski.color.rgb2lab(goodpalette)[0]
    lab_bad_palette2 = ski.color.rgb2lab(AnotherBadpalette)[0]
    lbuart = pm.getArtistPalette('Abovetheclouds610.png')

    Badpalettefeatures = fio.feature_extraction(lbuart, lab_bad_palette)
    Goodpalettefeatures = fio.feature_extraction(lbuart, lab_good_palette)
    AnotherBadpalettefeatures = fio.feature_extraction(lbuart, lab_bad_palette2)
    prod_features = [Goodpalettefeatures, Badpalettefeatures, AnotherBadpalettefeatures]


    palettea = '/home/er/Downloads/ProjectPyTest/DESIGNSEEDS/DESIGNSEEDS/palettes/AbstractColor.png'

    lab_designer_palette = ski.color.rgb2lab(
                                ski.io.imread(palettea))
    lbfirstrow = pm.get_palette_first_row(lab_designer_palette)
    lbu = pm.get_unique_palette_colors(lbfirstrow)
    kmeans = cm.findimagekmeans('AbstractColor.png')

    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(features, scores)
    predicted_scores = lasso_model.predict(prod_features)
    print("Predicted: "+str(predicted_scores))

    distanceft = scorer.calculateDistance(lbuart, lab_bad_palette2)
    distancen = scorer.calculateDistance(lbuart, lab_good_palette)
    distancef = scorer.calculateDistance(lbuart, lab_bad_palette)

    print(distancen)
    print(distancef)
    print(distanceft)
    maxdistance = 181/predicted_scores[0]
    print((distancef-distancen)/maxdistance)
    print((predicted_scores[0]-predicted_scores[1])/1)

if __name__ == '__main__':
    test_model()
