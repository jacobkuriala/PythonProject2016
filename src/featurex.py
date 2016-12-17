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
    return fxh.calculate_mean(Rcolors), fxh.calculate_mean(Gcolors), fxh.calculate_mean(Bcolors)

""""""""""""""""""""""""
"    Pixel coverage    "
""""""""""""""""""""""""
def extract_Lcov(designer_palette, generated_palette):
    """
    Extract the lightness coverage

    """
    designer_palette_lightness = designer_palette[:, 0]

    generated_palette_lightness = generated_palette[:, 0]

    range_i = np.amax(designer_palette_lightness) - np.amin(designer_palette_lightness)
    range_c = np.amax(generated_palette_lightness) \
              - np.amin(generated_palette_lightness)

    return range_c / range_i

def extract_Scov(designer_palette, generated_palette):
    """
    Extract the saturation coverage

    """
    hsv_designer_palette = skimage.color.rgb2hsv(
                            skimage.color.lab2rgb([designer_palette]))
    designer_palette_saturation = hsv_designer_palette[:, :, 1].flatten()

    hsv_generated_palette = skimage.color.rgb2hsv(
                            skimage.color.lab2rgb([generated_palette]))
    generated_palette_saturation = hsv_generated_palette[:, :, 1].flatten()

    range_i = np.amax(designer_palette_saturation) - np.amin(designer_palette_saturation)
    range_c = np.amax(generated_palette_saturation) \
              - np.amin(generated_palette_saturation)

    return range_c / range_i

def extract_soft_recoloring_error(designer_palette, generated_palette,
                                  equal):
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

def extract_min_color_dist(generated_palette):
    return min(fxh.calculate_distances(generated_palette))

def extract_max_color_dist(generated_palette):
    return max(fxh.calculate_distances(generated_palette))

def extract_mean_color_dist(generated_palette):
    return fxh.calculate_mean(fxh.calculate_distances(generated_palette))
