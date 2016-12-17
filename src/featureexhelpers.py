import numpy as np

#These functions generally will take 1-d arrays and calculate a measure


def calculate_mean(l):
    """

    :param l: iterable of numbers
    :return: mean of numbers
    """

    if len(l)>0:
        return float(sum(l)/len(l))
    else:
        return 0.0


def calculate_euclid_dist(a, b):
    """
    Calculate the euclidean distance
    between two vectors
    """
    return np.linalg.norm(a - b)


def calculate_upc(p, c, k_means_palette):
    error = 0.0
    for j in k_means_palette:
        error += (calculate_euclid_dist(p, c) / calculate_euclid_dist(p, j))**2
    return 1 / error

def calculate_distances(palette):
    distances = []
    rows = palette.shape[0]
    if rows != 1:
        for i in range(0, rows):
            for j in range(i+1, rows):
                dist = calculate_euclid_dist(palette[i], palette[j])
                distances.append(dist)
    else:
        distances.append(0.0)

    return distances

def formatter(img):
    [list(pixel) for row in img for pixel in row]
    return
