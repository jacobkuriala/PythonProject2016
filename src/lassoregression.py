"""
Module contains lasso regression model that uses features from palettes
to predict scores
"""
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import scorer

import featurex as fx
from featureio import read_features

# Imports used if extracting features from artist palette in Data/palettes
# import pallette_manager as pm
# from configuremanager import Configures


def get_data():
    """
    Gets feature data and scores for each image from features.txtfiles

    Returns:
    A list of palette and its associated scores
    """

    feature_dict = read_features()
    palette_feature_list = []
    palette_score_list = []
    for palette_name, feature_list in feature_dict.items():
        palette_feature_list.append(feature_list)
        palette_score_list.append(scorer.getscore(palette_name))

    return palette_feature_list, palette_score_list


def train_model():
    """
    Trains model by using 60% of the features and their score in sample
    data
    Returns:
    Returns trained lasso regression model and test data
    """
    features, scores = get_data()
    X_train, X_test, y_train, y_test = train_test_split(features, scores,
                                                        test_size=0.4,
                                                        random_state=0)

    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(X_train, y_train)

    return lasso_model, X_test, y_test


def test_model():
    """
    Tests model by using 40 percent of samples data. Evaluates model using
    coefficient of Determination. It later creates a plot graph that depicts
    predicted vs actual values
    """
    model, x_test, y_test = train_model()
    predicted_scores = model.predict(x_test)
    print("Predicted Scores: " + predicted_scores)
    print("Coefficient of Determination: " + str(model.score(x_test, y_test)))

    # Create Graph
    figure, axis = plt.subplots()
    axis.scatter(y_test, predicted_scores)
    axis.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--',
              lw=4)
    axis.tick_params(axisis='x', labelsize=10)
    axis.tick_params(axisis='y', labelsize=10)
    axis.set_xlabel('Actual', fontsize=18)
    axis.set_ylabel('Predicted', fontsize=18)
    axis.set_title('All features', fontsize=25)
    plt.show()


def production_model(original_palette):
    """
    Trains the model by using all of the samples data. Then it tests it by
    extracting features from a new designer palette match with two bad palettes
    and a good palette. Finally it prints out the predictions for each palette
    and coefficient of determination.
    Args:
    original_palette = original_palette from A
    """
    features, scores = get_data()
    # Three palettes inspired in the original_palette
    bad_palette1 = np.array([[[0.0, 0.0, 0.0], [163.0, 73.0, 164.0],
                              [205.0, 16.0, 26.0], [103.0, 152.0, 114.0],
                              [63.0, 72.0, 204.0]]], dtype=np.uint8)
    bad_palette2 = np.array([[[225.0, 225.0, 0.0], [0.0, 255.0, 255.0],
                              [255.0, 0.0, 255.0], [0.0, 255.0, 64.0],
                              [255.0, 0.0]]], dtype=np.uint8)
    good_palette = np.array([[[221.0, 135.0, 210.0], [135.0, 169.0, 221.0],
                              [135.0, 221.0, 156.0], [218.0, 221.0, 135.0],
                              [49.0, 59.0, 79.0]]], dtype=np.uint8)

    lab_bad_palette1 = ski.color.rgb2lab(bad_palette1)[0]
    lab_good_palette = ski.color.rgb2lab(good_palette)[0]
    lab_bad_palette2 = ski.color.rgb2lab(bad_palette2)[0]
    lab_og_p = original_palette  # pm.getArtistPalette('Abovetheclouds610.png')

    # Extract features for each original palette/good bad palette combination
    prod_features = [fx.feature_extraction(lab_og_p, lab_bad_palette1),
                     fx.feature_extraction(lab_og_p, lab_bad_palette2),
                     fx.feature_extraction(lab_og_p, lab_good_palette)]

    # Train lasso regression model with all sample data
    lasso_model = linear_model.Lasso(alpha=0.1)
    lasso_model.fit(features, scores)

    # Predict scores for two bad palettes and the good one
    predicted_scores = lasso_model.predict(prod_features)
    print("Predicted: "+str(predicted_scores))


if __name__ == '__main__':
    test_model()
