'''
Created on Aug 12, 2015

@author: Polina

This module computes matching and distance between 2 sets of points (color clusters)

Munkres algorithm (also called the Hungarian algorithm or the Kuhn-Munkres algorithm)
is used to find the
minimum weighted distance of a bipartite graph. This is nothing but a minimum
distance between the matrix k x k as evaluated by the Munkres algorithm; where
k is the parameter of k-means algorithm. Munkres algorithm returns bipartite
paring that are the indexes of the matrix constructed by the random image's
code points and the training image's code points. The code points of the random
image are compared with all the code points of the training set and a minimum
distance counter is measured.

Below are the dependencies which are required to be satisfied before
running this module:

1. This module uses the Munkres module. This library can be installed as below:
pip3 install munkres3

'''
from munkres import Munkres

from Chameleon.nearest_palette import create_distance_matrix, calculate_distance


def match_and_calculate_distance(test_codes, colors):  
    result = create_distance_matrix(test_codes, colors)
    mymunk = Munkres()
    bipartite_pairings = mymunk.compute(result)
    # print(filename)
    potential_distance = calculate_distance(bipartite_pairings, result)
    #print('potential_distance, current_distance', potential_distance, current_distance)
    return potential_distance