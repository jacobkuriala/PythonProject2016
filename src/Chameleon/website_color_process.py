# -*- coding: utf-8 -*-
'''
Created on Oct 31, 2014

@author: David
@author: Anshul

This file is designed to replace the css of a website. First, an image
of the user's choice is chosen. Then, using nearest_palette, that image
is compared against a training set of images that returns a training image
with color clusters that have the least distance to the uploaded image.

Once this image is obtained, the css to be modified is opened and two
collections are considered:
  * Dictionary of element names to color values
  * unique hex colors

After these are obtained, the palette of the training image is sorted
into dark and light colors, using a threshold on the lightness property.
The color is in LAB form. More colors are generated if its palette is
smaller than the amount of unique colors in the CSS. Dark colors are considered
background if there are more light colors than dark. Otherwise, light colors
are considered background.

Finally, dark and light elements are replaced according to the background and
non background elements and a CSS with new colors is produced.
'''
from __future__ import print_function

import os
import sys

sys.path.insert(0, os.path.abspath('.'))
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, AdobeRGBColor, HSLColor
from Chameleon import nearest_palette
import re
import shutil
import zipfile

sys.path.append('/usr/lib/python3.4/')
sys.path.append("/usr/local/lib/python3.4/dist-packages/")

import codecs
def color_hex_finder(css_file):
    '''
    Uses a regular expression to find elements and colors in
    the given css_file.

    Args:
        css_file: The filename of a css file
    Returns:
        list(color_palette): A list of unique colors found in the css
        element_list: A dictionary mapping of element name to hex color
    '''
    #file_p = open(css_file)
    file_p = codecs.open(css_file, 'r', encoding='latin-1')
    fulltext = file_p.read()
    return color_hex_finder_text(fulltext)


def color_hex_finder_text(fulltext):
    '''
    Uses a regular expression to find elements and colors in
    the given css_file.

    Args:
        css_file: The filename of a css file
    Returns:
        list(color_palette): A list of unique colors found in the css
        element_list: A dictionary mapping of element name to hex color
    '''
    # patt = re.compile(r"""
    # ([\s]{4,}(?P<element>[A-z\-]*?):   # Matches the element's name
    # .*?                             # Skips any space beyond the :
    # (?P<color>[#][a-fA-F0-9]+)[;])  # Reads the color in hex format
    # """, re.X)
    #PV - this has to be fixed.
    patt = re.compile(r"""
    ((?P<element>[A-z\-]*?):     # Matches the element's name
    [\s]*?                          # Skips any space beyond the :
    (?P<color>[#][a-fA-F0-9]+))  # Reads the color in hex format
    """, re.X)
    #print('pattern',patt)
    # Iterate through entire string
    element_list, color_palette = color_hex_finder_pattern(fulltext, patt)
    # Print results
    # print("Color palette: ", color_palette)
    # print("element list: ", element_list)

    # print("Second pattern with =")
    #can be = instead of : <td bgcolor="#e0ffff">
    patt = re.compile(r"""
    ((?P<element>[A-z\-]*?)=     # Matches the element's name
    [\s]*?                          # Skips any space beyond the :
    (?P<color>["][#][a-fA-F0-9]+["]))  # Reads the color in hex format
    """, re.X)
    #print('pattern',patt)
    # Iterate through entire string
    element_list_, color_palette_ = color_hex_finder_pattern(fulltext, patt)

    # Print results
    # print("Color palette: ", color_palette_)
    # print("element list: ", element_list_)
    return list(color_palette)+list(color_palette_) , element_list+element_list_

def color_hex_finder_pattern(fulltext, patt):
    element_list = []
    color_palette = set()
    for res in patt.finditer(fulltext):
        hex_= res.group("color")
        col = hex_
        #print(col)
        if len(hex_) == 4:
            col =  hex_[1] * 2 + hex_[2] * 2 + hex_[3] * 2 #'#' 
        if (([res.group("element"), col])) not in element_list:
            element_list.append([res.group("element"), col])
        if col not in color_palette:
            color_palette.add(col)
    return element_list, color_palette


#TODO: ADD NAMED AN HSL, HSLA colors
def color_rgba_finder(css_file):
    '''
    Uses a regular expression to find elements and colors in
    the given css_file.

    Args:
        css_file: The filename of a css file
    Returns:
        list(color_palette): A list of unique colors found in the css
        element_list: A dictionary mapping of element name to rgba color
    '''
    file_p = codecs.open(css_file, 'r', encoding='latin-1')
    fulltext =   file_p.read()
    #print(fulltext)
    # patt = re.compile(r"""
    # ((?P<element>[A-z\-]*?):     # Matches the element's name
    # [\s]*?                          # Skips any space beyond the :
    # (?P<color>(rgba\()[.,0-9]+(\))))  # Reads the color in rgba format rgba(244,244,244,0.99)
    # """, re.X) #(rgba\()[0-9,.]+(\))
    patt =  re.compile(r"""((?P<element>[A-z\-]*?)[\"]*?[\s]*?:[\"]*?[\s]*?(?P<color>(rgba\()[.,0-9]+(\))))""", re.X)
    #print('pattern',patt)
    color_palette, element_list = color_rgba_finder_pattern(fulltext, patt)

    patt =  re.compile(r"""((?P<element>[A-z\-]*?)[\"]*?[\s]*?[=][\"]*?[\s]*?(?P<color>(rgba\()[.,0-9]+(\))))""", re.X)
    #print('pattern',patt)
    color_palette_, element_list_ = color_rgba_finder_pattern(fulltext, patt)
    return list(color_palette)+list(color_palette_), element_list +element_list_

def color_rgba_finder_pattern(fulltext, patt):
    # Iterate through entire string
    element_list = []
    color_palette = set()
    for res in patt.finditer(fulltext):
        rgba= res.group("color")
        colstring = rgba.replace('rgba(', '')  
        colstring = colstring.replace(')', '')  
        collist = colstring.split(',')
        #print(rgba,colstring , collist)
        col = (int(collist[0]), int(collist[1]), int(collist[2]),float(collist[3]))
        if (([res.group("element"), col])) not in element_list:
            element_list.append([res.group("element"), col])
        if col not in color_palette:
            color_palette.add(col)
    # Print results
    # print("Color palette: ", color_palette)
    # print("element list: ", element_list)
    return color_palette, element_list
def color_generator(list_colors, original_number_of_colors):
    '''
    Adds a color to a list of colors. The list may be of dark or light
    colors. Uses the count of colors to decide which color, in linear order,
    to edit.
    Args:
        list_colors: A list of LAB values in the format [L, A, B]
        original_number_of_colors: How many colors were in list_colors before
        colors were generated to the end of it
    Returns:
        list(list_colors): Modified by reference.
    '''
    if len(list_colors) == 0:
        return
    #print('in color_generator', len(list_colors), original_number_of_colors)
    current_element = len(list_colors) % original_number_of_colors
    color_to_modify = (current_element
                       + len(list_colors) // original_number_of_colors) - 1
    scalar = 10
    #print('in color_generator', len(list_colors), original_number_of_colors, current_element, color_to_modify)
    # Check threshold to decide which direction to go
    if list_colors[color_to_modify][0] < 50:
        scalar *= -1

    # Is it possible to bring color further in one direction?
    if 0 <= list_colors[color_to_modify][0] + scalar <= 100:
        list_colors.append([list_colors[color_to_modify][0] + scalar,
                            list_colors[color_to_modify][1],
                            list_colors[color_to_modify][2]])

    else:
        pass #print('list_colors[color_to_modify][0] + scalar <0 or >100')

def color_generator_mod(list_colors, original_number_of_colors, threshold, step):
    '''
    Adds a color to a list of colors. The list may be of dark or light
    colors. Uses the count of colors to decide which color, in linear order,
    to edit.
    Args:
        list_colors: A list of LAB values in the format [L, A, B]
        original_number_of_colors: How many colors were in list_colors before
        colors were generated to the end of it
    Returns:
        list(list_colors): Modified by reference.
    Mod version is more careful in choice of step and threshold
    '''
    if len(list_colors) == 0:
        return
    print('in color_generator', len(list_colors), original_number_of_colors)
    current_element = len(list_colors) % original_number_of_colors
    color_to_modify = (current_element
                       + len(list_colors) // original_number_of_colors) - 1

    # Check threshold to decide which direction to go
    if list_colors[color_to_modify][0] < threshold:
        step *= -1

    # Is it possible to bring color further in one direction?
    if 0 <= list_colors[color_to_modify][0] + step <= 100:
        list_colors.append([list_colors[color_to_modify][0] + step,
                            list_colors[color_to_modify][1],
                            list_colors[color_to_modify][2]])

    else:
        print('list_colors[color_to_modify][0] + scalar <0 or >100')


def discover_k_best_palettes(folder, image_name, k, restrict =0):
    '''
    Opens up a file based on a parameter, calls clustering from the nearest
    palette library, and converts the colors from the resulting image into
    sRGB colors.

    Args:
        image_name: Name of the image to discover the palette of
    Returns:
        pal_clr_list_lab: A list of [L, A, B] values
    '''
    # Obtain filename that has nearest palette to a test input
    print (folder + '/' + image_name)
    result = []
    prevmatches =[] #to avoid getting same match due to rounding
    color_points = open("backend/cropbox.txt").read()
    mindist =0
    #to get only palettes with specified # of colors
    restrict = 0
    if restrict>0:
        done =0
        while not done:
            res = nearest_palette.find_K_nearest_clusters(folder + '/' + image_name, mindist, prevmatches)
            result_filename, img_cluster_lab, img_cluster_centr_lab, dist = res
            print(len(img_cluster_lab))
            if(len(img_cluster_lab)==restrict):
                done =1
                print('done')
                pal_clr_list_lab=get_palette(color_points, result_filename)
                result.append((img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist))
            else:
                print('not done')
                prevmatches.append(result_filename)
    else:        
        for i in range(0,k):
            print('getting', i, 'th palette, dist =', mindist)
            res = nearest_palette.find_K_nearest_clusters(folder + '/' + image_name, mindist, prevmatches)
            if len(res)>0:
                result_filename, img_cluster_lab, img_cluster_centr_lab, dist = res
                prevmatches.append(result_filename)
                pal_clr_list_lab=get_palette(color_points, result_filename)
                result.append((img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist))
                mindist = dist
            else:
                print("No match was found")
    return result


def discover_palette(folder, image_name, ignore_weights):
    '''
    Opens up a file based on a parameter, calls clustering from the nearest
    palette library, and converts the colors from the resulting image into
    sRGB colors.

    Args:
        image_name: Name of the image to discover the palette of
        ignore_weights: depending on this flag, palettes are wighted or unweighted
    Returns:
        pal_clr_list_lab: A list of [L, A, B] values
    '''
    # Obtain filename that has nearest palette to a test input
    #print (folder + '/' + image_name)
    if ignore_weights:
        result_filename, img_cluster_lab, img_cluster_centr_lab, dist = nearest_palette.find_nearest_cluster(folder + '/' + image_name)
        color_points = open("backend/cropbox.txt").read()
        pal_clr_list_lab=get_palette(color_points, result_filename)
        return img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist
    else:
        result_filename, img_cluster_lab, img_cluster_centr_lab, dist = weighted_palette.find_nearest_cluster_weighted(folder+'/'+image_name)
        # color_points = open("backend/cropbox.txt").read()
        # pal_clr_list_lab=get_palette(color_points, result_filename)
        return img_cluster_lab, img_cluster_centr_lab, result_filename, img_cluster_centr_lab, dist



def get_palette(color_points, result_filename):    
    # Obtain start and end points of RGB; save it to string
    start_index = color_points.find(result_filename)
    start_index = color_points.find('\n', start_index) + 1
    end_index = color_points.find('\n', start_index)
    rgb_color_str = color_points[start_index:end_index]
    #print(rgb_color_str)
    pal_color_list = []

    # Convert string into list of RGB tuples
    rgb_patt = re.compile("[(](?P<r>.*?)[,][ ](?P<g>.*?)[,][ ](?P<b>.*?)[)]")
    #print(rgb_color_str)
    for res in rgb_patt.finditer(rgb_color_str):
        try:
            pal_color_list.append((int(res.group("r")),
                                   int(res.group("g")),
                                   int(res.group("b"))))
        except:
            print(res.group("b"))
    #print(pal_color_list) # really rgb
    #render_palette(pal_color_list, 'palette_rgb', 'rgb')
    # Convert tuples of letters to RGBColor
    pal_color_list_rgb = [AdobeRGBColor(r, g, b, True)
                          for r, g, b in pal_color_list]
    #print('pal_color_list_rgb', pal_color_list_rgb)
    pal_clr_list_lab = [convert_color(color, LabColor)
                        for color in pal_color_list_rgb]
    pal_clr_list_lab = [[x.lab_l, x.lab_a, x.lab_b]
                        for x in pal_clr_list_lab]
    #print('pal_clr_list_lab:', pal_clr_list_lab)
    return pal_clr_list_lab


def lightness_separator(lab_colors):
    '''
    Takes in a list of LABcolor objects and returns two lists. The first list
    contains dark colors; the second list contains light colors. This is
    determined by an arbitrary threshold of 50 for the lightness aspect of LAB.

    Args:
        lab_colors: A list of LAB values in the format [L, A, B]
    Returns:
        dark_color_list: A list of colors determined to dark by threshold
        light_color_list: A list of colors determined to be light by threshold
    '''
    dark_color_list = list()
    light_color_list = list()

    # Separate dark colors from light using palette from cluster image
    for color in lab_colors:
        if color[0] > 50:
            light_color_list.append(color)
        else:
            dark_color_list.append(color)

    # If no dark colors, pick the darkest from light colors and move it there
    if len(dark_color_list) == 0:
        darkest = light_color_list[0]
        for light in light_color_list:
            darkest = darkest if darkest[0] > light[0] else light
        light_color_list.remove(darkest)
        dark_color_list.append(darkest)
        dark_color_list[0][0] -= 30

    # If no light colors, pick the lightest from dark colors and move it there
    elif len(light_color_list) == 0:
        lightest = dark_color_list[0]
        for dark in dark_color_list:
            lightest = dark if lightest[0] > dark[0] else lightest
        dark_color_list.remove(lightest)
        light_color_list.append(lightest)
        light_color_list[0][0] += 30

    return dark_color_list, light_color_list

def create_swap_dict_mod(PAL_color_list_LAB, unique_css_HEX, use_mod_generator):
    '''
    Picks a color from the unique_css_HEX and maps it to a color that currently exists
    in the elem_color_pairs.

    Args:
        elem_color_pairs: A dictionary mapping of element name to hex color
        unique_css_HEX: Color list to be used for  elements
    Returns:
        matching: Dictionary for swapping current color 
    '''
    #this is list of (orig, new)
    matching = []
    new_color_list ={}
    original_number_of_pal_colors = len(PAL_color_list_LAB)

    step=0
    if use_mod_generator:
        step = 100/(len(unique_css_HEX)-original_number_of_pal_colors)
    # Map elements to either dark colors or light colors,
    # depending on if it is a background element or not.
    # number is not matching elem_color_pairs because only unique colors are picked
    for color in unique_css_HEX:
        change_made = False
        for clr in PAL_color_list_LAB:
            if len(new_color_list) ==0 or clr in tuple(new_color_list.values()):
                continue
            else:
                new_color_list[color] = clr
                change_made = True
                matching.append((color, clr))
                break
        if not change_made:

            if use_mod_generator:
                color_generator_mod(PAL_color_list_LAB,
                                original_number_of_pal_colors, 50, step)
            else:
                color_generator(PAL_color_list_LAB,
                            original_number_of_pal_colors)

            clr = PAL_color_list_LAB[len(PAL_color_list_LAB) - 1]
            new_color_list[color] = clr
            change_made = True
            matching.append((color, clr))

    print('matching', len(matching))
    return matching

def create_swap_dict(elem_color_pairs, most_color_list, less_color_list):
    '''
    Picks a color from the most_color_list or the less_color list, depending
    on criteria described below, and maps it to a color that currently exists
    in the elem_color_pairs.

    Args:
        elem_color_pairs: A dictionary mapping of element name to hex color
        most_color_list: Color list to be used for non-background elements
        less_color_list: Color list to be used for background elements
    Returns:
        background_css: Dictionary for swapping current background color to
        color from less_color_list
        non_background_css: Dictionary for swapping current background element
        to color from most_color_list
    '''
    background_css = dict()
    non_background_css = dict()
    #this is list of (orig, new)
    matching = []
    original_number_of_less_colors = len(less_color_list)
    original_number_of_most_colors = len(most_color_list)

    # Map elements to either dark colors or light colors,
    # depending on if it is a background element or not.
    # number is not matching elem_color_pairs because only unique colors are picked
    for element_name, color in elem_color_pairs:
        change_made = False
        if "background" in element_name:
            # Use a less color_set as a background
            if color in background_css.values():
                continue
            for clr in less_color_list:
                if clr in tuple(background_css.values()):
                    continue
                else:
                    background_css[color] = clr
                    change_made = True
                    matching.append((color, clr))
                    break
            if not change_made:
                color_generator(less_color_list,
                                original_number_of_less_colors)
                clr = less_color_list[len(less_color_list) - 1]
                background_css[color] = clr
                change_made = True
                matching.append((color, clr))
        else:
            # Use a majority color as a non background element
            if color in non_background_css.values():
                continue
            for clr in most_color_list:
                if clr in tuple(non_background_css.values()):
                    continue
                else:
                    non_background_css[color] = clr
                    change_made = True
                    matching.append((color, clr))
                    break
            if not change_made:
                color_generator(most_color_list,
                                original_number_of_most_colors)
                clr = most_color_list[len(most_color_list) - 1]
                non_background_css[color] = clr
                change_made = True
                matching.append((color, clr))
    print('matching', len(matching))
    print(' background_css ,  non_background_css ', len(background_css), len(non_background_css))
    return background_css, non_background_css, matching


def write_css(name_of_file, set_of_css_colors,
              back_color_swap, non_back_color_swap):
    '''
    This function swaps the css colors with the generated palatte colors.
    This will replace and write new colors to the css file.

    Args:
        name_of_file: The name of the CSS file to swap the colors into
        set_of_css_colors: A dictionary mapping of element name to hex color
        back_color_swap: Dictionary for swapping current background color to
        color from less_color_list
        non_back_color_swap: Dictionary for swapping current background element
        to color from most_color_list
 '''
    # Open original file and open a file that will contain the swapped colors
    print(name_of_file)
    filetext = open(name_of_file).readlines()
    file_p = open(name_of_file, "w")
    print ('opening for writing', name_of_file)
    for line in filetext:
        elem_found = False
        for elem in set_of_css_colors:
            if elem_found:
                break
            # Letters made uppercase to avoid capitalization misses
            elif elem.upper() in line.upper():
                # Process element
                temp_lab = []
                try:
                    if "background".upper() in line.upper():
                        temp_lab = LabColor(*back_color_swap[elem])
                    else:
                        temp_lab = LabColor(*non_back_color_swap[elem])
                    temp_rgb = convert_color(temp_lab, AdobeRGBColor)
                    temp_rgb.rgb_r = abs(temp_rgb.rgb_r)
                    temp_rgb.rgb_g = abs(temp_rgb.rgb_g)
                    temp_rgb.rgb_b = abs(temp_rgb.rgb_b)

                    print("Swapping", elem, "with", temp_rgb.get_rgb_hex())
                    file_p.write(line.replace(elem, temp_rgb.get_rgb_hex()))
                    elem_found = True
                except:
                    print('error converting to LAB', elem)


        # If the element is not present, write it back the way it was.
        if not elem_found:
            file_p.write(line)

def signal_handler(signum, frame):
    raise Exception("Timed out!")

def find_css(temp_folder, save_copy):
    """
    This function goes into tmp folder and return all css and html file paths.
    Args:
        temp_folder: Auto generated temp folder path.
    Returns:
        css_file_paths: A list of full paths for the .css files.
        html_file_paths: A list of full paths for the .html files.
    """
    css_file_paths = []
    html_file_paths = []
    for root, dirs, files in os.walk(temp_folder):
        for f in files:
            if (f.endswith(".css") or f.endswith(".svg")) and not f.startswith('_'):
                css_file_paths.append(os.path.join(root, f))
            if f.endswith(".html"):
                html_file_paths.append(os.path.join(root, f))
            #print("Legit html") 


    #print(css_file_paths)
    if save_copy:
        #html
        for p in html_file_paths:
            #print("Legit css", f)
            #save original css files - in case we need to recolor
            #in the same dir but starting with _
            dest =os.path.join(os.path.dirname(p),'_'+os.path.basename(p))
            #print('before css copy', p , ' to ', dest)
            shutil.copy2(p, dest)  
        # dest =os.path.join(temp_folder,'_index.html')
        # shutil.copy2(os.path.join(temp_folder,'index.html'), dest)  
        for p in css_file_paths:
            #print("Legit css", f)
            #save original css files - in case we need to recolor
            #in the same dir but starting with _
            dest =os.path.join(os.path.dirname(p),'_'+os.path.basename(p))
            #print('before css copy', p , ' to ', dest)
            shutil.copy2(p, dest)  
            # in_file = open(p)
            # indata = in_file.read()

            # out_file = open(dest, 'w')
            # out_file.write(indata)

            # out_file.close()
            # in_file.close()      
            #print('after css copy')
            # sys.stdout.flush()
    # print('css_file_paths: ', css_file_paths)
    # print('html_file_paths: ', html_file_paths)
    return css_file_paths, html_file_paths

def find_original_css(temp_folder):
    """
    This function goes into tmp folder and return all css file paths THAT START WITH _.
    Args:
        temp_folder: Auto generated temp folder path.
    Returns:
        css_file_paths: A list of full paths for the .css files.
    """
    css_file_paths = []
    for root, dirs, files in os.walk(temp_folder):
        for f in files:
            if f.endswith(".css") and f.startswith('_'):
                css_file_paths.append(os.path.join(root, f))
    return css_file_paths

def find_image(temp_folder):
    """temp_folder is the full path"""
    #image is directly in tmp folder
    images = []
    for f in os.listdir(temp_folder):
        if f.endswith('.png') or  f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.gif'):
            images.append(f)#(os.path.join(temp_folder, f))
    return images

def matching_algorithm_original(PAL_color_list_LAB, result_elems, result_folder):
    """
    If there are more darker colors than light,
    use the darker colors as non-background elements.
    """
    dark_colors, light_colors = lightness_separator(PAL_color_list_LAB)
    render_palette(dark_colors, result_folder+'/dark_colors', 'LAB')
    render_palette(light_colors, result_folder+'/light_colors', 'LAB')
    if len(dark_colors) >= len(light_colors):
        swap_background_css, swap_non_background_css, matching = \
            create_swap_dict(result_elems, dark_colors, light_colors)
    else:
        swap_background_css, swap_non_background_css, matching = \
            create_swap_dict(result_elems, light_colors, dark_colors)
    #print("Background CSS mapping:", swap_background_css)
    #print("Non background CSS mapping:", swap_non_background_css)
    render_matching(sorted(matching, key=lambda x:x[0]), result_folder+'/final_matching_orig_alg', 'hexlab')
    render_matching(sorted(matching, key=lambda x:x[1][0]), result_folder+'/final_matching_orig_alg_LAB', 'hexlab')
    # render_matching(swap_background_css, 'swap_background_css', 'hexlab')
    # render_matching(swap_non_background_css, 'swap_non_background_css', 'hexlab')
    return swap_background_css, swap_non_background_css

def matching_algorithm_original_modified(PAL_color_list_LAB, unique_css_HEX, result_folder):
    """
    same as original, but without using background/foreground elements to separate color set
    clustering?
    try with no sepratation just use their expansion routine
    """
    use_mod_generator=0

    matching = \
        create_swap_dict_mod(PAL_color_list_LAB, unique_css_HEX, use_mod_generator)
    if use_mod_generator:
        render_matching(sorted(matching, key=lambda x:x[0]), result_folder+'/final_matching_mod_gen', 'hexlab')
        render_matching(sorted(matching, key=lambda x:x[1][0]), result_folder+'/final_matching_mod_gen_LAB', 'hexlab')
    else:
        render_matching(sorted(matching, key=lambda x:x[0]), result_folder+'/final_matching_mod', 'hexlab')
        render_matching(sorted(matching, key=lambda x:x[1][0]), result_folder+'/final_matching_mod_LAB', 'hexlab')
    return matching

def matching_algorithm_original_modified1(PAL_color_list_LAB, unique_css_HEX):
    """
    same as original, but without using background/foreground elements to separate color set
    clustering?
    try with no sepratation just use their expansion routine
    """

    matching = \
        create_swap_dict_mod(PAL_color_list_LAB, unique_css_HEX)
    render_matching(sorted(matching, key=lambda x:x[0]), 'final_matching_mod1', 'hexlab')
    render_matching(sorted(matching, key=lambda x:x[1][0]), 'final_matching_mod1_LAB', 'hexlab')
    return matching


_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'

def rgb(triplet):
    try:
        triplet = triplet.replace('#', '')
        #print('good', triplet)
        return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]
    except:
        print('Error in hex to rgb',triplet)

def triplet(rgb, lettercase=LOWERCASE):
    return format(rgb[0]<<16 | rgb[1]<<8 | rgb[2], '06'+lettercase)

def hex_to_rgb(hex):
    #print('{}, {}'.format(rgb('aabbcc'), rgb('AABBCC')))
    # -> (170, 187, 204), (170, 1rgb87, 204)

    # print('{}, {}'.format(triplet((170, 187, 204)),
    #                       triplet((170, 187, 204), UPPERCASE)))
    # # -> aabbcc, AABBCC

    # print('{}, {}'.format(rgb('aa0200'), rgb('AA0200')))
    # # -> (170, 2, 0), (170, 2, 0)

    # print('{}, {}'.format(triplet((170, 2, 0)),
    #                       triplet((170, 2, 0), UPPERCASE)))
    return rgb(hex)


def check_LAB_boundaries(new_color, pal_color):
    if(new_color[0]<0 or new_color[0]>100):
        #print('Bad L:', new_color)
        new_color[0]=pal_color.lab_l
        # if new_color[0]<0:
        #     new_color[0]=0
        # if new_color[0]>100:
        #     new_color[0]=100
    if(abs(new_color[1])>127):
        print('Bad a:', new_color)
        if new_color[1]>127:
            new_color[1]=127
        if new_color[1]<-127:
            new_color[1]=-127
    if(abs(new_color[2])>127):
        print('Bad b:', new_color)
        if new_color[2]>127:
            new_color[2]=127
        if new_color[2]<-127:
            new_color[2]=-127
    #print('new_color', new_color)
    #render_palette([new_color], str(new_color), 'LAB')
    return new_color

def matching_algorithm_new(PAL_color_list_LAB, unique_css_HEX, L_only, results_folder, render_all=False):
    """
    PAL_color_list_LAB is the list of palette colors LAB
    [[85.2787927101103, 1.1833082428296948, 4.302894776701693],
    unique_css_HEX is HEX colors from .css
    results_folder is for rendering

    LAB is only needed for finding closest image
    Use HSL to manipulate brightness
    """
    # print('PAL_color_list_LAB', PAL_color_list_LAB)
    # print('unique_css_HEX', unique_css_HEX)

    num_clusters = 0
    done = 0
    big_set = []
    small_set = []
    big_set_clustered =[]
    css_is_smallest = 0
    palette_is_smallest = 0
    # render DS palette
    render_palette_vertical(PAL_color_list_LAB, os.path.join(results_folder,'DS_palette'), 'LAB')
    #check which set is bigger and cluster bigger set colors 
    #into num_clusters clusters,
    #where num_clusters is the number of colors in the smaller set
    if len(PAL_color_list_LAB) > len(unique_css_HEX):
        print('len(PAL_color_list_LAB) > len(unique_css_HEX)')
        num_clusters = len(unique_css_HEX)
        css_is_smallest = 1
    elif len(PAL_color_list_LAB) < len(unique_css_HEX):
        print('len(PAL_color_list_LAB) < len(unique_css_HEX)')
        num_clusters = len(PAL_color_list_LAB)
        palette_is_smallest = 1
    else:
        done = 1
    if not done:
        print('num_clusters, palette_is_smallest, css_is_smallest', num_clusters, palette_is_smallest, css_is_smallest)
        print('convert HEX to LAB')
        UNIQUE_CSS_LAB = []
        #UNIQUE_CSS_RGB = []
        ORIG_HEX_TO_LAB = {}
        for h in unique_css_HEX:
            #get RBG and convert
            rgb =hex_to_rgb(h)
            #UNIQUE_CSS_RGB.append(rgb)
            RGB = AdobeRGBColor(rgb[0]/ 255, rgb[1]/ 255, rgb[2]/ 255)
            lab = convert_color(RGB, LabColor)
            UNIQUE_CSS_LAB.append(lab)
            ORIG_HEX_TO_LAB[lab] = h
        print('LEN UNIQUE_CSS_LAB', len(UNIQUE_CSS_LAB))
   
        #render_palette(UNIQUE_CSS_LAB, results_folder+'/orig_css_conv_to_LAB', 'LAB')
        #render_palette(UNIQUE_CSS_RGB, 'orig_css_conv_to_rgb', 'rgb')
        #print('UNIQUE_CSS_LAB', UNIQUE_CSS_LAB)
        lablist =[]
        import numpy
        if palette_is_smallest:
            #print('palette smallest', UNIQUE_CSS_LAB)
            big_set = UNIQUE_CSS_LAB
            small_set = PAL_color_list_LAB
            lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in big_set])
        if css_is_smallest:
            #print('css smallest', PAL_color_list_LAB)
            big_set = PAL_color_list_LAB
            small_set = UNIQUE_CSS_LAB
            lablist = big_set
        if not palette_is_smallest and not css_is_smallest:
            print('WTF')
        #print('PAL_color_list_LAB', PAL_color_list_LAB)
        print('clustering bigger set into ', num_clusters)
        import scipy.cluster
       
        #big_set_clustered contains k colors representing k clusters.
        ###todo: how do we know which color in which cluster???
        big_set_clustered, dist = scipy.cluster.vq.kmeans(lablist, num_clusters)
        #print('big_set_clustered',big_set_clustered)
        #print('dist', dist)
    else:
        print('same length - just match')
        big_set_clustered = big_set
    render_palette(big_set_clustered, results_folder+'/big_set_clustered', 'LAB')
    #render_palette(big_set, results_folder+'/big_set', 'LAB')
    print('Actual Matching Procedure')
    matching ={} #palette LAB : css LAB 
    #Match clusters to the colors in the smaller list.
    #find matching using Hungarian algorithm
    from Chameleon.nearest_palette import create_distance_matrix
    from munkres import Munkres
    result = create_distance_matrix(small_set, big_set_clustered)
    mymunk = Munkres()
    bipartite_pairings = mymunk.compute(result)
    print('bipartite_pairings', bipartite_pairings)
    lab_big_set_clustered =[]
    for b in big_set_clustered:
        lab = LabColor(b[0], b[1], b[2])
        lab_big_set_clustered.append(lab)
    lab_small_set=[]
    for b in small_set:
        lab = LabColor(b[0], b[1], b[2])
        lab_small_set.append(lab)
    # print('lab_big_set_clustered', lab_big_set_clustered)
    # print('lab_small_set', lab_small_set)
    for s,b in bipartite_pairings:
        #print(b,s)
        if palette_is_smallest:
            # print('small_set[s]', small_set[s])
            # print('big_set_clustered[b]', big_set_clustered[b])
            matching[lab_small_set[s]] = b #lab_big_set_clustered[b]
            #matching.append(small_set[s], lab_big_set_clustered[b])
        if css_is_smallest:
            matching[lab_small_set[b]]=lab_big_set_clustered[c]            
            #matching.append(small_set[b],lab_big_set_clustered[c])
    #For each Cluster i, C[i], find a color c[j] in C[i] that is closest to j by L value. 
    #print('matching', matching)
    if palette_is_smallest:
        print('Creating additional colors')
        #this is matching of representative css to palette - doesn't work in this case
        #need to find actual original css hex that matched to palette
        #get all colors in cluster, and find closest to representative
        from cluster import euclidean_distance

        #find color closest to centroid and map it to palette color

        closest_color_centroid ={}
        for centroid in lab_big_set_clustered:
            mindist =float('inf')
            closest_color =0
            for color in big_set:
                #print([centroid[0], centroid[1], centroid[2]], color)
                dist =euclidean_distance([centroid.lab_l, centroid.lab_a, centroid.lab_b], [color.lab_l, color.lab_a, color.lab_b] )
                if mindist>dist:
                    mindist=dist
                    closest_color = color
            #calc vector between closest color and centroid - to be applied to all other colors in cluster
            #vector = numpy.array([0, 0, 0])
            # we don't want same vector for all colors!!!!
            vector = numpy.subtract(numpy.array([closest_color.lab_l, closest_color.lab_a, closest_color.lab_b]),  numpy.array([centroid.lab_l, centroid.lab_a, centroid.lab_b]))
            #print(' vector',vector, numpy.array([closest_color.lab_l, closest_color.lab_a, closest_color.lab_b]),  numpy.array([centroid.lab_l, centroid.lab_a, centroid.lab_b]))
            closest_color_centroid[lab_big_set_clustered.index(centroid)] = (closest_color, vector)
            
        #print('closest_color_centroid', closest_color_centroid)
        for k, v in matching.items():
            matching[k]= closest_color_centroid[v][0]
        #print('matching', len(matching), matching)
        #print('centroid', [v for k,v in closest_color_centroid.items()])
        #print('big set', big_set)
        render_matching(matching, results_folder+'/matching_clustered', 'LAB')
        
        #match L to css even for original palette colors
        if L_only==2:
            for orig, new in matching.items():
                #print('orig.lab_l, new.lab_l', orig.lab_l, new.lab_l)
                #orig.lab_l = new.lab_l
                #use HSL
                #print('orig LAB', orig)
                orig.lab_l =  match_brightness_HSL(new, orig)
                #print('after HSL LAB', orig)
                #print('orig.lab_l, new.lab_l', orig.lab_l, new.lab_l)
            render_matching(matching, results_folder+'/matching_clustered_L', 'LAB')
        

        cluster_mapping=[]
        for color in big_set:
            if color not in [v[0] for k,v in closest_color_centroid.items()]:
                #A vector v belongs to cluster i if it is closer to centroid i than any other centroids
                mindist =float('inf')
                closest_centroid =[]
                for centroid in lab_big_set_clustered:
                    dist =euclidean_distance([centroid.lab_l, centroid.lab_a, centroid.lab_b], [color.lab_l, color.lab_a, color.lab_b])
                    if mindist>dist:
                        mindist=dist
                        closest_centroid = centroid
                cluster_mapping.append((lab_big_set_clustered.index(closest_centroid), color))
            # else:
            #     print('color in [v[0] for k,v in closest_color_centroid.items()]')
        # crentroids they obviously map to themselves
        render_matching([(lab_big_set_clustered[i], c) for i, c in sorted(cluster_mapping,key=lambda x: x[0])], results_folder+'/cluster_mapping', 'LAB')

        # print('cluster_mapping', cluster_mapping)
        # print(len(matching), len(cluster_mapping))
        # For every other color C[i]: i<>j create a new color by applying a vector c[j]->C[i] to smaller color. 
        #print(matching)
        extended_palette =[]
        for centroid, unmapped_color in cluster_mapping:
            big_set_color, vector = closest_color_centroid[centroid]
            #add diff between current color and closest color to centroid
            additionalvector = numpy.subtract(numpy.array([unmapped_color.lab_l, unmapped_color.lab_a, unmapped_color.lab_b]), numpy.array([big_set_color.lab_l, big_set_color.lab_a, big_set_color.lab_b]))
            pal_color = [s for s,b in matching.items() if b==big_set_color][0]
            #print('big_set_color, pal_color', big_set_color, pal_color)
            #print(' vector',vector)
            if L_only==1:
                new_color=numpy.array( [pal_color.lab_l, pal_color.lab_a, pal_color.lab_b])
                new_color[0]= pal_color.lab_l+ numpy.add(vector, additionalvector)[0]

            elif L_only==2:
                new_color_lab = pal_color
                new_color_lab.lab_l = match_brightness_HSL(unmapped_color, new_color_lab)
                new_color=numpy.array( [new_color_lab.lab_l, new_color_lab.lab_a, new_color_lab.lab_b])
                #render_matching([(unmapped_color, new_color_lab)], 'test_final_'+str(unmapped_color.lab_l), 'LAB')
                #new_color[0]= unmapped_color.lab_l #match L don't change

            else:
                new_color = numpy.add(numpy.array( [pal_color.lab_l, pal_color.lab_a, pal_color.lab_b]), numpy.add(vector, additionalvector))
            if  new_color[0]>100:
                print(pal_color, 'pal_color \n',vector, additionalvector)

            new_color =check_LAB_boundaries(new_color, pal_color)

            # print('palette', numpy.array( [pal_color.lab_l, pal_color.lab_a, pal_color.lab_b]))
            # print('new_color', new_color)
            extended_palette.append((pal_color, lab_small_set.index(pal_color), LabColor(new_color[0], new_color[1], new_color[2])))
            #print('unmapped_color, new_color',unmapped_color, new_color)
            # if LabColor(new_color[0], new_color[1], new_color[2]) in matching.keys():
            #     print('WTH it already exists!!!!')
            matching[LabColor(new_color[0], new_color[1], new_color[2])] = unmapped_color
    render_matching([(a,c) for a,b,c in sorted(extended_palette, key=lambda x: x[1])], results_folder+'/extended_palette', 'LAB')
    #print('extended_palette', extended_palette)
    if css_is_smallest:
        print('no additional colorsA vector v belongs to cluster i if it is closer to centroid i than any other centroids')

    #print('matching', len(matching))
    #final matching: orig hex(css) to new hex (palette)
    #add #
    #print('ORIG_HEX_TO_LAB', ORIG_HEX_TO_LAB)
    final_matching ={}
    final_matching_rgb ={}
    for new, orig in matching.items():
        #print('orig',orig)
        rgb = convert_color(new, AdobeRGBColor)
        rgb = rgb.get_upscaled_value_tuple()
        #print('rgb', rgb)       
        #new_hex =rgb.get_rgb_hex() #'#%02x%02x%02x' % (rgb['rgb_r'],rgb['rgb_g'],rgb['rgb_b'])
        #new_hex ='#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255)
        if ORIG_HEX_TO_LAB[orig] in final_matching.keys():
            print('WTH already exists in final_matching!')
        final_matching[ORIG_HEX_TO_LAB[orig]]=rgb #new_hex

        #final_matching_rgb[rgb]=rgb
    #print('final_matching' , final_matching)

    if(len(final_matching)!=len(big_set)):
        print('Final matching has wrong number of colors!', len(final_matching), len(big_set))
    if L_only==1:
        print('L_only==1')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[0]), results_folder+'/final_matching_Lonly', 'hexrgb')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder+'/final_matching_Lonly1', 'hexrgb')
        render_separate_palettes(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder, 'hexrgb')
        render_matching( final_matching, results_folder+'/final_matching', 'rgb')
    elif L_only==2:
        print('L_only==2')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[0]), results_folder+'/final_matching_Lmatch', 'hexrgb')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder+'/final_matching_Lmatch1', 'hexrgb')
        render_separate_palettes(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder, 'hexrgb')
        #print(final_matching_rgb)
        #render_matching( final_matching_rgb, results_folder+'/final_matching_Lmatch_rgb', 'rgb')
    else:
        print('Some other L')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[0]), results_folder+'/final_matching', 'hexrgb')
        render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder+'/final_matching_1', 'hexrgb')
        render_separate_palettes(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder, 'hexrgb')
    #print('results_folder', results_folder)
    return final_matching



def match_brightness_HSL(LabOrig, LabToMatch):
    """LabOrig, LabToMatchh have to be in LabColor format
    Returns L that has to be assigned"""
    #convert to HSL
    #L is from this one
    #render_matching([(LabOrig, LabToMatch)], 'test'+str(LabToMatch.lab_l), 'LAB')
    #print(LabOrig.lab_l, LabToMatch.lab_l)
    HslOrig = convert_color(LabOrig, HSLColor)
    #update this one

    HslToMatch = convert_color(LabToMatch, HSLColor)
    #print('hsl orig:', HslToMatch.hsl_l, HslOrig.hsl_l)
    HslToMatch.hsl_l= HslOrig.hsl_l
    #print('hsl after:', HslToMatch.hsl_l, HslOrig.hsl_l)
    LabToMatch = convert_color(HslToMatch, LabColor)
    #print('Lab after:', LabOrig.lab_l, LabToMatch.lab_l)
    #render_matching([(LabOrig, LabToMatch)], 'test_after'+str(LabToMatch.lab_l), 'LAB')
    return LabToMatch.lab_l


def test_color_matching(tmpfolder, image_name, option):
    """tmpfolder is a fully qualified path.
    imagename is name of a file directly in tmpfolder"""
    img_cluster_lab, PAL_color_list_LAB, result_filename, img_cluster_centr_lab, dist = discover_palette(tmpfolder, image_name)
    css_names, html_names = find_css(tmpfolder, 1)#"../uploads/example_website2/css/agency.css"       # For example website 2
    unique_css_HEX = []
    result_elems = []

    css_name = ''
    for css in css_names:
        css_name = css
        _unique_css_HEX, _result_elems = color_hex_finder(css)
        unique_css_HEX.extend(_unique_css_HEX)
        result_elems.extend(_result_elems)
    print('_result_elems', len(_result_elems))
    #print(sorted(_result_elems, key=lambda x: x[1]))
    print('unique_css_HEX', len(unique_css_HEX))
    if option =='new':
        L_only=1
        matching=matching_algorithm_new(PAL_color_list_LAB, unique_css_HEX, L_only, '')
        recolor_css_new(matching, css_names, tmpfolder)
    if option =='orig':
        matching_algorithm_original(PAL_color_list_LAB, result_elems ,'')
    if option =='mod':
        matching_algorithm_original_modified(PAL_color_list_LAB, unique_css_HEX ,'')


# def get_palette_and_css(webfolder, imagefolder, image_name):
#     """
#     prep step for all matching and recoloring
#     tmpfolder and image_name are fully qualified paths"""
#     img_cluster_lab, PAL_color_list_LAB, result_filename, PAL_img_clusters_lab, dist = discover_palette(imagefolder, image_name)
#     css_names, html_names = find_css(webfolder)
#     unique_css_HEX = []
#     result_elems = []
#     css_name = ''
#     for css in css_names:
#         css_name = css
#         _unique_css_HEX, _result_elems = color_hex_finder(css)
#         unique_css_HEX.extend(_unique_css_HEX)
#         result_elems.extend(_result_elems)
#     return img_cluster_lab, PAL_color_list_LAB, result_filename, unique_css_HEX, result_elems, css_names, PAL_img_clusters_lab

def get_css_palette(webfolder, imagefolder, image_name):
    """
    prep step for all matching and recoloring
    tmpfolder and image_name are fully qualified paths"""
    print("in get_css_palette")
    #img_cluster_lab, PAL_color_list_LAB, result_filename, PAL_img_clusters_lab, dist = discover_palette(imagefolder, image_name)
    css_names, html_names = find_css(webfolder, 0)
    html_name ='index.html'
    #index.html could have inline styles
    found =0
    for name in html_names:
        if( 'index.html' in name):
            css_names.append(os.path.join(webfolder,name))
            found  =1
            break
    if found ==0:
        print('html_names: no index', html_names)
        html_name=html_names[0]
        css_names.append(html_names[0])
    print('found css', css_names)
    unique_css_HEX = []
    result_elems = []
    unique_css_rgba = []
    result_elems_rgba = []
    css_name = ''
    for css in css_names:
        css_name = css
        print(css_name)
        try:
            _unique_css_HEX, _result_elems = color_hex_finder(css)
            # print("_unique_css_HEX ", _unique_css_HEX)
            # print("_result_elems", _result_elems)
            _unique_css_rgba, _result_elems_rgba = color_rgba_finder(css)
            # print("_unique_css_rgba ", _unique_css_rgba)
            # print("_result_elems_rgba", _result_elems_rgba)
            unique_css_HEX.extend(_unique_css_HEX)
            result_elems.extend(_result_elems)
            unique_css_rgba.extend(_unique_css_rgba)
            result_elems_rgba.extend(_result_elems_rgba)
        except:
            print('Problem getting colors from ', css_name)

    print('len(unique_css_HEX)', len(unique_css_HEX))
    return  unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba

def recolor_css(tmpfolder, image_name, option):
    """tmpfolder is a fully qualified path.
    imagename is name of a file directly in tmpfolder"""
    img_cluster_lab, PAL_color_list_LAB, result_filename, dist = discover_palette(tmpfolder, image_name)
    #print 'Got palette:', PAL_COL"../static/out/"+str(tmpfoldername)+result_filenameOR_LIST_LAB
    # Obtain palette of unique colors and dictionary
    # css_name = "example_website/css/freelancer.css"  # For example website 1
    css_names, html_names = find_css(tmpfolder, 1)#"../uploads/example_website2/css/agency.css"       # For example website 2
    unique_css_HEX = []
    result_elems = []
    #temp fix
    css_nammae = ''
    for css in css_names:
        css_name = css
        _unique_css_HEX, _result_elems = color_hex_finder(css)
        unique_css_HEX.extend(_unique_css_HEX)
        result_elems.extend(_result_elems)
    print('_result_elems', len(_result_elems))
    print(sorted(_result_elems, key=lambda x: x[1]))
    print('unique_css_HEX', len(unique_css_HEX))
        #PV: for now hardcoded main - but need to handle multiple
        # if css.endswith("startbootstrap.css"):
        #     break
    # #also removing #!
    # unique_css_extend = list()
    # for x in unique_css_HEX:
    #     x2 = x
    #     if len(x) == 4:
    #         x2 =  x[1] * 2 + x[2] * 2 + x[3] * 2 #'#' +
    #     if x2.replace('#', '') not in unique_css_extend:
    #         unique_css_extend.append(x2.replace('#', ''))
    # print('unique_css_extend: ', len(unique_css_extend))
    # #render_palette(unique_css_extend, 'original_css_hex_original_matching', 'hex')

    swap_background_css, swap_non_background_css = matching_algorithm_original(PAL_color_list_LAB, _result_elems)

    #copy palette and original image to static folder
    curdir  = os.getcwd()
    fullpath = curdir+"/backend/DESIGNSEEDS/palettes/"+result_filename
    #print('dir', curdir, fullpath)
    tmpfoldername = os.path.basename(os.path.normpath(tmpfolder)) 
    print(tmpfoldername, 'tmpfoldername')
    shutil.copyfile(fullpath, curdir+"/static/out/"+str(tmpfoldername)+"/"+result_filename)
    print(tmpfolder+'/'+image_name)
    shutil.copyfile(tmpfolder+'/'+image_name, curdir+"/static/out/"+str(tmpfoldername)+"/"+image_name)
    print(css_name)
    # Open original file and open a file that will contain the swapped colors
    write_css(css_name, unique_css_HEX,
              swap_background_css, swap_non_background_css)
    #write_multiple_css(css_names, unique_css_HEX, swap_background_css, swap_non_background_css)
    print ('DONE RECOLORING')
    return  "../static/out/"+str(tmpfoldername)+"/"+result_filename

def recolor_css_old(swap_background_css, swap_non_background_css, set_of_css_colors, css_names, orig_dir):
    # Open original file and open a file that will contain the swapped colors
    for css in css_names:
        print(css)
        filetext = open(orig_dir+os.path.basename(css)).readlines()
        file_p = open(css, "w")
        for line in filetext:
            elem_found = False
            for elem in set_of_css_colors:
                if elem_found:
                    break
                # Letters made uppercase to avoid capitalization misses
                elif elem.upper() in line.upper():
                    # Process element
                    temp_lab = []
                    try:

                        if elem in swap_background_css:#"background".upper() in line.upper():
                            temp_lab = LabColor(*swap_background_css[elem])
                        elif elem in swap_non_background_css:
                            temp_lab = LabColor(*swap_non_background_css[elem])

                        temp_rgb = convert_color(temp_lab, AdobeRGBColor)
                        
                        temp_rgb.rgb_r = abs(temp_rgb.rgb_r)
                        temp_rgb.rgb_g = abs(temp_rgb.rgb_g)
                        temp_rgb.rgb_b = abs(temp_rgb.rgb_b)
                    except:
                        print('error converting to LAB', elem, temp_lab)
                    try:
                        print("Swapping", elem, "with", temp_rgb.get_rgb_hex())
                        file_p.write(line.replace(elem, temp_rgb.get_rgb_hex()))
                        elem_found = True
                    except:
                        print('error swapping', elem)
            # If the element is not present, write it back the way it was.
            if not elem_found:
                file_p.write(line)


def recolor_css_new(matching, active_css_elem, orig_dir):
    '''
    This function swaps the css colors with the generated palatte colors.
    This will replace and write new colors to the css files.

    Args:
        matching - hex to rgb, orig : new
        
        orig_css_names - copied files containing original colors
        active_css - css files names and cleaned contents to write to
    '''
    # Open original file and open a file that will contain the swapped colors
    print('matching', matching)
    print(active_css_elem)
    for _css in active_css_elem:
        css =''
        filetext=''
        if len(_css) ==2:
            css, filetext=_css
        else:
            css = _css
        print(css)
        # filetext = open(css).readlines()#open(os.path.join(orig_dir,os.path.basename(css))).readlines()
        file_p = open(css, "w")

        for line in filetext:
            elem_found = False
            for orig, rgb in matching.items():
                #new = '#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255)
                new = '#%02x%02x%02x' % (rgb[0],rgb[1],rgb[2])
                # Letters made uppercase to avoid capitalization misses
                #try 3-letter hexes too!
                if orig[0]=='#':
                    orig3 = '#'+orig[1]+orig[3]+orig[5]
                else:
                    orig3 =  '#'+orig[0]+orig[2]+orig[4]
                print(orig3, orig)

                if orig.upper() in line.upper():
                    # Process element
                    file_p.write(line.replace(orig, new))
                    print('replaced', orig, new)
                    elem_found = True
                elif orig3.upper() in line.upper():
                    # Process element
                    file_p.write(line.replace(orig3, new))
                    elem_found = True
                    print('replaced', orig3, new)
            if elem_found==False:
                file_p.write(line)
        file_p.close()

def zip_code_css(tmpfolder):
    """ into 
    zip dir called code inside tmpfolder into code.zip
    """
    zipf = zipfile.ZipFile(tmpfolder+'/code.zip', 'w')
    tmpfolder = tmpfolder+"/code/"
    for root, dirs, files in os.walk(tmpfolder):
        for f in files:
            zipf.write(os.path.join(root, f))
    zipf.close()

def unzip_code_css(filename, tmpfolder):
    """ into dir called code inside tmpfolder
    """
    fh = open(filename, 'rb')
    z = zipfile.ZipFile(fh)
    z.extractall()
    # for name in z.namelist():
    #     try:
    #         outfile = z.open(name, 'wb')
    #         outfile.write(os.path.join(tmpfolder, 'code',z.read(name)))
    #         outfile.close()
    #     except:
    #         print('Cannot unzip', name)
    fh.close()

import tarfile
def unzip_tar(filename, tmpfolder):
    """ into dir called code inside tmpfolder
    """ 
    #'w4.tar.gz', '/home/linka/python/autoimage_flask/uploads/tmpbea6oz3l/code'
    # open the tarfile and use the 'r:gz' parameter
    # the 'r:gz' mode enables gzip compression reading
    tfile = tarfile.open(filename, 'r:gz')
    # 99.9% of the time you just want to extract all
    # the contents of the archive.
    tfile.extractall(tmpfolder)
    tfile.close()



def new_matching_test(tmpfolder, image_name):
    """tmpfolder is a fully qualified path. imagename is name of a file directly in tmpfolder"""
    img_cluster_lab, PAL_color_list_LAB, result_filename, dist = discover_palette(tmpfolder, image_name)
    #print 'Got palette:', PAL_color_list_LAB
    # Obtain palette of unique colors and dictionary
    # css_name = "example_website/css/freelancer.css"  # For example website 1
    css_names, html_names = find_css(tmpfolder, 1)#"../uploads/example_website2/css/agency.css"       # For example website 2
    unique_css_HEX = []
    result_elems = []
    #temp fix
    css_name = ''
    for css in css_names:
        css_name = css
        _unique_css_HEX, _result_elems = color_hex_finder(css)
        unique_css_HEX.extend(_unique_css_HEX)
        result_elems.extend(_result_elems)
        #PV: for now hardcoded main - but need to handle multiple

    # print('len(result_elems) == len(unique_css_HEX)', len(result_elems) == len(unique_css_HEX))
    # print('result_elems: ', len(result_elems))
    # print('result_elems: ', result_elems, '\n')

    # print('unique_css_HEX: ', len(unique_css_HEX))
    # print('unique_css_HEX: ', unique_css_HEX , '\n')

    #also removing #!
    # unique_css_extend = list()
    # for x in unique_css_HEX:
    #     x2 = x
    #     if len(x) == 4:
    #         x2 =  x[1] * 2 + x[2] * 2 + x[3] * 2 #'#' +
    #     if x2.replace('#', '') not in unique_css_extend:
    #         unique_css_extend.append(x2.replace('#', ''))

    print('unique_css_HEX: ', len(unique_css_HEX))
    render_palette(unique_css_HEX, 'original_css_hex', 'hex')
    #print('unique_css_extend: ', unique_css_extend , '\n')
    
    render_palette(PAL_color_list_LAB, 'test_original_palette', 'LAB')
    final_matching=matching_algorithm_new(PAL_color_list_LAB, unique_css_HEX, results_folder)
    #write to file
    # import pickle
    # dmfile = open(tmpfolder+'/matching', 'wb')
    # pickle.dump(final_matching,dmfile)
    # dmfile.close()
    # # Open original file and open a file that will contain the swapped colors
    # write_multiple_css(css_name, unique_css_HEX,
    #           swap_background_css, swap_non_background_css)
    # print 'DONE REOCLORING'
    return PAL_color_list_LAB, result_filename, final_matching

def write_multiple_css(name_of_files, set_of_css_colors,
              back_color_swap, non_back_color_swap):
    '''
    This function swaps the css colors with the generated palatte colors.
    This will replace and write new colors to the css files.

    Args:
        name_of_file: The names of the CSS file to swap the colors into
        set_of_css_colors: A dictionary mapping of element name to hex color
        back_color_swap: Dictionary for swapping current background color to
        color from less_color_list
        non_back_color_swap: Dictionary for swapping current background element
        to color from most_color_list
 '''
    # Open original file and open a file that will contain the swapped colors
    
    for css in name_of_files:
        print(css)
        filetext = open(css).readlines()
        file_p = open(css, "w")
        print ('opening for writing', css)
        for line in filetext:
            elem_found = False
            for elem in set_of_css_colors:
                if elem_found:
                    break
                # Letters made uppercase to avoid capitalization misses
                elif elem.upper() in line.upper():
                    # Process element
                    temp_lab = []
                    try:
                        if "background".upper() in line.upper():
                            temp_lab = LabColor(*back_color_swap[elem])
                        else:
                            temp_lab = LabColor(*non_back_color_swap[elem])
                        temp_rgb = convert_color(temp_lab, AdobeRGBColor)
                        temp_rgb.rgb_r = abs(temp_rgb.rgb_r)
                        temp_rgb.rgb_g = abs(temp_rgb.rgb_g)
                        temp_rgb.rgb_b = abs(temp_rgb.rgb_b)
    
                        print("Swapping", elem, "with", temp_rgb.get_rgb_hex())
                        file_p.write(line.replace(elem, temp_rgb.get_rgb_hex()))
                        elem_found = True
                    except:
                        print('error converting to LAB', elem)
    
    
            # If the element is not present, write it back the way it was.
            if not elem_found:
                file_p.write(line)


def render_palette1(palette, filename, mode):
    if len(palette)>0:
        #palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
        size = [200, 100*len(palette)]
        #print('size', size)
        im = Image.new('LAB', size, color=0)
        draw = ImageDraw.Draw(im)
        y =0
        for k in palette:
            draw.rectangle([(0,y), (200, y+100)], fill=(int(k[0]), int(k[1]), int(k[2])) , outline=(255,255,255))
            y =y+100
        del draw
        im.save(filename, "lab")
    else:
        print('0 length palette!')


def render_palette(palette, filename, mode):
    """ render horizonltally, fix image size, 
    vary size of color buckets to fit image size"""
    if len(palette)>0:
        print(filename, mode)
        #palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
        size = [500, 100]
        bucket_size = 500/len(palette)
        #print('size', size, len(palette), bucket_size)
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        """[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
        outline – Color to use for the outlin. fill – Color to use for the fill."""
        x =0
        y = 100 #height
        for k in palette:
            rgb =()
            if mode =='rgb':
                rgb =k
            if mode=='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)
            if mode=='LAB':
                _rgb =()
                if (type(k)==LabColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    try:
                        _rgb =convert_color(LabColor(k[0], k[1], k[2]), AdobeRGBColor)
                    except:
                        print('Error converting lab to RGB:', k)

                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted LAB to RBG:', rgb)
                #rgb = AdobeRGBColor(_rgb[0], _rgb[1], _rgb[2], True)
                #print(rgb)
                #_hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            if mode=='HSL':
                _rgb =()
                if (type(k)== HSLColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    print('HSL: bad format')
                    break
                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted HSL to RBG:', rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            draw.rectangle([(x,0), (x+bucket_size, y)], fill=rgb, outline=(255,255,255))
            #draw.rectangle([(0,y), (200, y+100)], fill=rgb , outline=(255,255,255))
            x =x+bucket_size
        del draw
        im.save(filename+".png", "PNG")
    else:
        print('0 length palette!')

def render_palette_vertical(palette, filename, mode):
    """ render horizonltally, fix image size, 
    vary size of color buckets to fit image size"""
    if len(palette)>0:
        #palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
        size = [100, 700]
        bucket_size = 700/len(palette)
        print('rendering', filename, mode,'size', size, len(palette), bucket_size)
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        """[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
        outline – Color to use for the outlin. fill – Color to use for the fill."""
        x =100
        y = 0 #height
        for k in palette:
            rgb =()
            if mode =='rgb':
                if(len(rgb)==3):
                    rgb =k
                else:
                    rgb = (k[0], k[1], k[2])
                #print(rgb)
            if mode=='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                elif len(k)==4:
                    rgb = (k[0], k[1], k[2])
                else:
                    rgb =hex_to_rgb(k)
            if mode=='LAB':
                _rgb =()
                if (type(k)==LabColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    try:
                        _rgb =convert_color(LabColor(k[0], k[1], k[2]), AdobeRGBColor)
                    except:
                        print('Error converting lab to RGB:', k)

                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted LAB to RBG:', rgb)
                #rgb = AdobeRGBColor(_rgb[0], _rgb[1], _rgb[2], True)
                #print(rgb)
                #_hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            if mode=='HSL':
                _rgb =()
                if (type(k)== HSLColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    print('HSL: bad format')
                    break
                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted HSL to RBG:', rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            draw.rectangle([(0,y), (100, y+bucket_size)], fill=rgb , outline=(255,255,255))
            #draw.rectangle([(0,y), (200, y+100)], fill=rgb , outline=(255,255,255))
            y =y+bucket_size
        del draw
        im.save(filename+".png", "PNG")
    else:
        print('0 length palette!')

def render_palette_vertical_weights(palette, filename, mode, height=500):
    """ 
    assuming that weight is a percentage!!!! rounded to int. like 50 (%)
    render horizonltally, fix image size, 
    vary size of color buckets to fit image size"""
    if len(palette)>0:
        #palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
        #size =[int(height/10), height] #[100, 1000]
        #print('palette', palette)
        totalweight = sum(w for k,w in palette)
        #print('totalweight', totalweight)
        bucket_size = int(height/totalweight) #10 #represents 1% len(palette)
        size =[int(height/10), bucket_size*totalweight] #[100, 1000]
        #print('bucket_size', bucket_size, size)
        #print('rendering w', filename, mode,'size', size, len(palette), bucket_size)
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        """[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
        outline – Color to use for the outlin. fill – Color to use for the fill."""
        x =100
        y = 0 #height
        for k, w in palette:
            rgb =()
            if mode =='rgb':
                rgb =k
            if mode=='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)
            if mode=='LAB':
                _rgb =()
                if (type(k)==LabColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    try:
                        _rgb =convert_color(LabColor(k[0], k[1], k[2]), AdobeRGBColor)
                    except:
                        print('Error converting lab to RGB:', k)

                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted LAB to RBG:', rgb)
                #rgb = AdobeRGBColor(_rgb[0], _rgb[1], _rgb[2], True)
                #print(rgb)
                #_hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            if mode=='HSL':
                _rgb =()
                if (type(k)== HSLColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    print('HSL: bad format')
                    break
                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted HSL to RBG:', rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            draw.rectangle([(0,y), (100, y+bucket_size*w)], fill=rgb )#, outline=(255,255,255))#(0,0,0))#(255,255,255))
            #draw.rectangle([(0,y), (200, y+100)], fill=rgb , outline=(255,255,255))
            y =y+bucket_size*w
        del draw
        im.save(filename+".png", "PNG")
    else:
        print('0 length palette!')

def render_palette_horizontal_weights(palette, filename, mode, width=500):
    """ 
    assuming that weight is a percentage!!!! rounded to int. like 50 (%)
    render horizonltally, fix image size, 
    vary size of color buckets to fit image size"""
    if len(palette)>0:
        #palette =  ['e4b9c0', 'f7e1b5', 'faebcc']

        totalweight = sum(w for k,w in palette)
        #print('totalweight', totalweight)
        bucket_size = int(width/totalweight) #10 #represents 1% len(palette)
        size =[bucket_size*totalweight, int(width/10)] #[100, 1000]
        #print('bucket_size', bucket_size, size)
        #print('rendering w', filename, mode,'size', size, len(palette), bucket_size)
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        """[(x0, y0), (x1, y1)] or [x0, y0, x1, y1]. The second point is just outside the drawn rectangle.
        outline – Color to use for the outlin. fill – Color to use for the fill."""
        x =0
        y = 100 #
        for k, w in palette:
            rgb =()
            if mode =='rgb':
                rgb =k
            if mode=='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)
            if mode=='LAB':
                _rgb =()
                if (type(k)==LabColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    try:
                        _rgb =convert_color(LabColor(k[0], k[1], k[2]), AdobeRGBColor)
                    except:
                        print('Error converting lab to RGB:', k)

                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted LAB to RBG:', rgb)
                #rgb = AdobeRGBColor(_rgb[0], _rgb[1], _rgb[2], True)
                #print(rgb)
                #_hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            if mode=='HSL':
                _rgb =()
                if (type(k)== HSLColor):
                    _rgb =convert_color(k, AdobeRGBColor)
                else:
                    print('HSL: bad format')
                    break
                rgb = _rgb.get_upscaled_value_tuple()
                #print('Converted HSL to RBG:', rgb)
                # _hex = _rgb.get_rgb_hex()
                # if '#' in _hex:
                #     rgb =hex_to_rgb(_hex[1:])
                # else:
                #     rgb =hex_to_rgb(_hex)
            draw.rectangle([(x,0), (x+bucket_size*w, y)], fill=rgb , outline=(255,255,255))#(0,0,0))#(255,255,255))
            #draw.rectangle([(0,y), (200, y+100)], fill=rgb , outline=(255,255,255))
            x =x+bucket_size*w
        del draw
        im.save(filename+".png", "PNG")
    else:
        print('0 length palette!')


def render_matching(input_matching, filename, mode):
    print('rendering matching to', filename)
    if len(input_matching)>0:
        y =0
        #print(type(input_matching))
        if type(input_matching) ==dict:
            matching = [(k,v) for k, v in input_matching.items() ]
        if type(input_matching) ==list:
            matching = input_matching

        size = [400, 100*len(matching)]    
        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)

        for k, v in matching:
            rgb =()
            rgb1 =()
            if mode =='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)
                if '#' in v:
                    rgb1 =hex_to_rgb(v[1:])
                else:
                    rgb1 =hex_to_rgb(v)
            if mode =='LAB':
                rgb=LAB_to_RGB(k)         
                rgb1=LAB_to_RGB(v) 
            if mode =='LABRGB':
                rgb=LAB_to_RGB(k)  
                rgb1=v       
            if mode =='hexlab':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)     
                rgb1=LAB_to_RGB(v) 
            if mode =='rgb':
                if isinstance(k, tuple):
                    rgb =k
                else:
                    rgb =(k.rgb_r, k.rgb_g, k.rgb_b)
                    print(rgb)
                if isinstance(v, tuple):
                    rgb1 =v
                else:
                    rgb1 =(v.rgb_r, v.rgb_g, v.rgb_b)
                    print(rgb)
                
            if mode =='hexrgb':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)    
                rgb1 =v
            draw.rectangle([(0,y), (200, y+100)], fill=rgb, outline=(255,255,255))
            draw.rectangle([(201,y), (400, y+100)], fill=rgb1, outline=(255,255,255))
            y = y+100
        del draw
        im.save(filename+".png", "PNG")
    else:
        print('0 length matching!')


from PIL import Image, ImageDraw
def render_separate_palettes(input_matching, filename, mode):
    if len(input_matching)>0:
        x =0
        #print(type(input_matching))
        if type(input_matching) ==dict:
            matching = [(k,v) for k, v in input_matching.items() ]
        if type(input_matching) ==list:
            matching = input_matching

        size = [700, 100]
        bucket_size = 700/len(input_matching)

        im = Image.new('RGB', size, color=0)
        draw = ImageDraw.Draw(im)
        im1 = Image.new('RGB', size, color=0)
        draw1 = ImageDraw.Draw(im1)

        for k, v in matching:
            rgb =()
            rgb1 =()
            if mode =='hex':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)
                if '#' in v:
                    rgb1 =hex_to_rgb(v[1:])
                else:
                    rgb1 =hex_to_rgb(v)
            if mode =='LAB':
                rgb=LAB_to_RGB(k)         
                rgb1=LAB_to_RGB(v) 
            if mode =='hexlab':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)     
                rgb1=LAB_to_RGB(v) 
            if mode =='rgb':
                rgb =k
                rgb1 =v
            if mode =='hexrgb':
                if '#' in k:
                    rgb =hex_to_rgb(k[1:])
                else:
                    rgb =hex_to_rgb(k)    
                rgb1 =v
            draw.rectangle([(x,0), (x+bucket_size, 100)], fill=rgb, outline=(255,255,255))
            draw1.rectangle([(x,0), (x+bucket_size, 100)], fill=rgb1, outline=(255,255,255))
            x =x+bucket_size
        del draw
        del draw1
        im.save(os.path.join(filename,"original_pal.png"), "PNG")
        im1.save(os.path.join(filename,"out_pal.png"), "PNG")
    else:
        print('0 length matching!')

def LAB_to_RGB(k):
    _rgb =()
    if (type(k)==LabColor):
        _rgb =convert_color(k, AdobeRGBColor)
    else:
        _rgb =convert_color(LabColor(k[0], k[1], k[2]), AdobeRGBColor)
    rgb = _rgb.get_upscaled_value_tuple()
    # _hex = _rgb.get_rgb_hex()
    # if '#' in _hex:
    #     rgb =hex_to_rgb(_hex[1:])
    # else:
    #     rgb =hex_to_rgb(_hex)
    return rgb

import fnmatch
def find_file(folderpath, pattern):
    """in folder, find path to index.html"""
    #folderpath = "/home/linka/python/autoimage_flask/uploads/tmpc_g62371/code"
    matches = []
    for root, dirnames, filenames in os.walk(folderpath):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


# from processor import Processor
# def clean_css(location):
#     """
#      Return a copy of the original CSS
#      but with the selectors not found in the document(s) removed.
#      https://github.com/peterbe/mincss"""
#     p = Processor(optimize_lookup=True)

#     #location ='/home/linka/python/autoimage_flask/uploads/tmp9l7mlh19/code/'
#     css_file_paths, html_paths = find_css(location, 0) 
#     #print('html_paths', html_paths)
#     url= "index.html"#http://www.cs.fsu.edu/department/faculty/sudhir/"
#     p.process_local(html_paths, url, location)
#     #p.process(url)
#     #p.process()
#     #InlineResult -This is where the results are stored for inline CSS.
#     # print("INLINES ".ljust(79, '-'))
#     # don't think we need this part
#     # for each in p.inlines:
#     #     print("On line %s" % each.line)
#     #     print('- ' * 40)
#     #     print("BEFORE")
#     #     print(each.before)
#     #     print('- ' * 40)
#     #     print("AFTER:")
#     #     print(each.after)
#     #     print("\n")
#     #This is where the results are stored for all referenced links to CSS files. 
#     #i.e. from things like <link rel="stylesheet" href="foo.css">
#     active_css = []
#     active_css_elem = []
#     hex_colors = set()
#     #print("LINKS ".ljust(79, '-'))
#     for each in p.links: 
#         filename = os.path.basename(each.href)
#         filename = filename.replace('%3F', '?')
#         csspath = os.path.join(location, filename)
#         try:
#             #delete file if after comes out empty
#             if len(each.after) ==0:
#                 os.remove(csspath) 
#                 #print('Removed', csspath)
#             else:
#                 colors, elem = color_hex_finder_text(each.after)
#                 #print(colors)
#                 hex_colors.update(colors)
#                 if len(colors)>0:#no need to look at css that have no colors 
#                     #overwrite the css file - later when processed
#                     # f = open(csspath, 'w')
#                     # f.write(each.after)
#                     # f.close()
#                     #return list of used css
#                     active_css_elem.append((csspath, each.after))
#                     active_css.append(csspath)
#                     #print('Rewrote', csspath)

#         except:
#             print('No file', csspath)
#     #print(list(hex_colors))
#     return list(hex_colors), active_css, active_css_elem


import subprocess
def download_website(codefilename,tmpfolder):
    """
    download with timeout
    """
    #args = ['wget', '--no-directories', '--no-clobber', '--no-parent', '--convert-links', '--page-requisites', '-r',  '-p',  '-E',  '-e', 'robots=off', '-P', os.path.join(tmpfolder, 'code'), codefilename]
    args = ['wget', '--no-directories', '--no-clobber', '--no-parent', '--convert-links', '--page-requisites', '-E',  '-e', 'robots=off', '-P', os.path.join(tmpfolder, 'code'), codefilename]
    subprocess.call(args, timeout=60*5) #time out in 5 min


if __name__ == "__main__":
    print( 'Started')
    #sudo python3 backend/website_color_process.py /home/linka/python/autoimage_flask/uploads/tmpLAimmd dimlight_forest_by_ferdinandladera-d82wydc.jpg

    #tmpfolder = "/home/linka/python/autoimage_flask/uploads/code/"
    #codefilename='www.cs.fsu.edu'
    tmpfolder ="/home/linka/python/autoimage_flask/uploads/bg3/"
    get_css_palette(tmpfolder, '', '')

    # filename = 'w11.zip'
    # unzip_code_css(os.path.join(tmpfolder, 'code', filename), os.path.join(tmpfolder, 'code'))
    # print('url, downloading' )
    # try:
    #     if check_file_size(codefilename):
    #         #--no-verbose
    #         #error_code= os.system('wget '+codefilename + ' --limit-rate=200k --no-clobber --no-parent --convert-links --random-wait --page-requisites -r -p -E -e robots=off -P '+ os.path.join(tmpfolder, 'code'))
    #         download_website(codefilename,tmpfolder)
    #     else:
    #         message = 'Web page size> 1GB'
    # except:
    #     message = 'Failed downloading web site' 
    # print(message)
    # find_image(temp_folder)
    """if len(sys.argv) != 4:
        print("Usage: website_color_process.py path_to_tmpfolder image_name option{new/orig/mod}")
        sys.exit(1)
    test_tmp_folder = sys.argv[1]
    test_image_name = sys.argv[2]
    option = sys.argv[3]"""
    # rgb =hex_to_rgb('ffffff')
    # print(rgb)
    #test_image_name = (input("Enter the name of the test image: "))
    #recolor_css(test_tmp_folder,test_image_name )
    # palette =  ['e4b9c0', 'f7e1b5', 'faebcc']
    # render_palette(palette, '/home/linka/python/autoimage_flask/testing/palette.png', 'hex')

    """if test_function=='new':
        new_matching_test(test_tmp_folder,test_image_name )
    #render_matching({}) 
    if test_function=='orig':
        recolor_css(test_tmp_folder, test_image_name)"""

    #zip_code_css(test_tmp_folder)
    #test_color_matching('/home/linka/python/autoimage_flask/testing/2_recolor/sidhir/','piyush.jpg', 'new')
    #test_color_change()
    #LAB_to_RGB([ 38.98484047,53.55269216,4.48114514])
    #test_color_change_HSL()
    # folder ="/home/linka/python/autoimage_flask/uploads/test/"
    #find_css("/home/linka/python/autoimage_flask/uploads/test/code")
    #'w4.tar.gz', '/home/linka/python/autoimage_flask/uploads/tmpbea6oz3l/code'
    #unzip_tar('/home/linka/python/autoimage_flask/testing/2_recolor/input_web/w4.tar.gz', "/home/linka/python/autoimage_flask/uploads/tmp_fv4x3q7/code/")
    #zip_code_css(tmpfolder)
    #clean_css("http://deanhume.com/home/blogpost/automatically-removing-unused-css-using-grunt/6101")
    #clean_css("http://localhost/rerender/facebook/22/Polina")
    #clean_css('/home/linka/python/autoimage_flask/uploads/tmp9l7mlh19/code/')
    #color_rgba_finder('/home/linka/Desktop/recoloring results/cs fsu/inn_tmp24u9hw3p/code/index.html')
    #color_rgba_finder('/home/linka/python/autoimage_flask/uploads/code/code/index.html')
    print("Done")