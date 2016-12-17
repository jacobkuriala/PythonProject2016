import featureexhelpers as fxh
import skimage
import skimage.color
import numpy as np


def extract_RGBmean(sioimage):
    """

    :param sioimage: RGBImage
    :return: mean of numbers
    """
    Rcolors = sioimage[:, :, 0].flatten()
    Gcolors = sioimage[:, :, 1].flatten()
    Bcolors = sioimage[:, :, 2].flatten()
    return fxh.calculate_mean(Rcolors), fxh.calculate_mean(Gcolors),
    fxh.calculate_mean(Bcolors)

""""""""""""""""""""""""
"    Pixel coverage    "
""""""""""""""""""""""""


def extract_Lcov(designer_palette, generated_palette):
    """
    Extract the lightness coverage. Lightness coverage is the quotient between
    range_c and range_i.
    Range_c is the difference between the maximum lightness and the minimum
    lightness of colors in a algorithmically generated palette and
    Range_i is the difference between the maximum lightness and the minimum
    lightness of colors in a designer created generated palette.

    Args:
    designer_palette: designer created palette with colors in LAB space
    generated_palette: algorithmically generated palette with colors
                        in LAB space

    Returns:
    A float that is the quotient between range_c and range_i.
    """
    designer_palette_lightness = designer_palette[:, 0]

    generated_palette_lightness = generated_palette[:, 0]

    range_i = np.amax(designer_palette_lightness) - \
    np.amin(designer_palette_lightness)
    range_c = np.amax(generated_palette_lightness) - \
    np.amin(generated_palette_lightness)

    return range_c / range_i


def extract_Scov(designer_palette, generated_palette):
    """
    Extract the saturation coverage. Saturation coverage is the quotient between
    range_c and range_i. The colors of both palettes are converted into HSV in
    order to extract saturation from each color
    Range_c is the difference between the maximum saturation and the minimum
    saturation of colors in a algorithmically generated palette and
    Range_i is the difference between the maximum saturation and the minimum
    saturation of colors in a designer created palette.

    Args:
    designer_palette: designer created palette with colors in LAB space
    generated_palette: algorithmically generated palette with colors
                        in LAB space

    Returns:
    A float that is the quotient between range_c and range_i.
    """
    hsv_designer_palette = skimage.color.rgb2hsv(
                            skimage.color.lab2rgb([designer_palette]))
    designer_palette_saturation = hsv_designer_palette[:, :, 1].flatten()

    hsv_generated_palette = skimage.color.rgb2hsv(
                            skimage.color.lab2rgb([generated_palette]))
    generated_palette_saturation = hsv_generated_palette[:, :, 1].flatten()

    range_i = np.amax(designer_palette_saturation) - \
              np.amin(designer_palette_saturation)
    range_c = np.amax(generated_palette_saturation) - \
              np.amin(generated_palette_saturation)

    return range_c / range_i


def extract_soft_recoloring_error(designer_palette, generated_palette, equal):
    """
    Soft recoloring is the error of recoloring a designer created palette with
    the colors of an algorithmically generated palette. To calculate this
    error, we use the sum of the product between the errors squared and the upc
    squared.
    Error is the distance between a pixel in the designer palette and a pixel
    the algorithmically generated palette.
    Upc is 1 divided by the sum of the quotients of the distance between designer
    created palette pixel and a generated color pixel and the error
    between designer created palette pixel and all the pixels in the generated
    palette.

    Args:
    designer_palette: designer created palette
    generated_palette: algorithmically generated_palette
    equal: True if both palettes are the same

    Returns:
    A float value that contains the soft recoloring. 0 if palettes are equal

    """
    soft_error = 0.0
    if not equal:
        for p in designer_palette:
            inner_sum = 0.0
            for c in generated_palette:
                upc = fxh.calculate_upc(p, c, generated_palette)
                error = fxh.calculate_euclid_dist(p, c)
                inner_sum += (upc)**2 * (error)**2
            soft_error += inner_sum

    return soft_error

""""""""""""""""""""""""
"    Color diversity   "
""""""""""""""""""""""""


def extract_min_color_dist(palette):
    """
    Extracts the minimum distance of distances between colors in a palette

    Args:
    palette: a color palette in LAB space

    Returns:
    A float of the minimum distance of distances between colors in a palette
    """
    return min(fxh.calculate_distances(palette))


def extract_max_color_dist(palette):
    """
    Extracts the max distance of distances between colors in a palette

    Args:
    palette: a color palette in LAB space

    Returns:
    A float of the max distance of distances between colors in a palette
    """
    return max(fxh.calculate_distances(palette))


def extract_mean_color_dist(palette):
    """
    Extracts the mean distance of distances between colors in a palette

    Args:
    palette: a color palette in LAB space

    Returns:
    A float of the mean distance of distances between colors in a palette
    """
    return fxh.calculate_mean(fxh.calculate_distances(palette))

""""""""""""""""""""""""
" Extract all features "
""""""""""""""""""""""""


def feature_extraction(designer_palette, generated_palette, equal=False):
    """
    Extract lightness coverage, saturation coverage, soft recoloring error,
    minimum, maximum and distance between a designer palette colors and
    an algorithmically generated palette

    Args:
    designer_palette: designer created palette with colors in LAB space
    generated_palette: algorithmically generated palette with colors
                        in LAB space
    equal: True if you are using the same palette otherwise False

    Returns:
    A list of list of the features. Features are in floating point values

    """
    features = []
    features.append(extract_Lcov(designer_palette, generated_palette))
    features.append(extract_Scov(designer_palette, generated_palette))
    features.append(extract_soft_recoloring_error(designer_palette,
                                                  generated_palette, equal))
    features.append(extract_min_color_dist(generated_palette))
    features.append(extract_max_color_dist(generated_palette))
    features.append(extract_mean_color_dist(generated_palette))

    return features
