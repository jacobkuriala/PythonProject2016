"""These functions generally will take 1-d arrays and calculate a measure.
It also provides a formatter function to format incorrect formatted
palettes"""
import numpy as np


def calculate_mean(num_list):
    """
    Calculates the mean in a list

    Args:
    num_list: iterable of numbers

    Returns:
    mean of numbers in list
    """

    if len(num_list) > 0:
        return float(sum(num_list) / len(num_list))
    else:
        return 0.0


def calculate_euclid_dist(vector_a, vector_b):
    """
    Calculate the euclidean distance between vector_a and vector_a

    Args:
    vector_a: np array of floats
    vector_b: np array of floats

    Returns:
    Float with the values of the euclidean distance between the two vectors
    in an np array.
    """
    return np.linalg.norm(vector_a - vector_b)


def calculate_upc(designer_palette_pixel, generated_color, generated_palette):
    """
    Calculate the upc between vector_a and vector_a

    Args:
    designer_palette_pixel: pixel in designer generated palette
    generated_color: color of algorithmically generated palette
    generated_palette: algorithmically generated palette

    Returns:
    Float with the values of the euclidean distance between the two vectors
    in an np array.
    """
    error = 0.0
    for pixel in generated_palette:
        error += (calculate_euclid_dist(designer_palette_pixel,
                                        generated_color) /
                  calculate_euclid_dist(designer_palette_pixel, pixel))**2
    return 1 / error


def calculate_distances(palette):
    """
    Iterates over a palette calculating the euclidean distance between each
    color.

    Args:
    palette: palette in LAB space

    Returns:
    List of distances between colors in a palette
    """
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

