'''
Created on Sep 24, 2014

@author: Anshul
@author: David

This module navigates through a local directory and opens up an image for
processing. This module processes the image to separate out it's palette and
the corresponding image. The coordinates (left, upper, right, lower) of
the image and the palette are appended to a file. This file is called
"cropbox.txt" which basically stores the iamges's and it's palette's crop
points. The picture part of the image is saved to a folder named "slices" and
optionally we can save the palette for all the images to the folder called
"palette". Next, for each of the palette we have cropped we extract the palette
colors from it and create a new palette to visually test from the original
palette. We save these palettes to the "palette" folder for all the images as
we process one by one. Finally, all the three attributes: Image crop points,
Palette crop points and RGB colors for the palette are written for each of the
processed images.

To run this code one needs to resolve the following dependencies:
1. Install Pillow Library. To install pillow use the following command:
pip3 install pillow

2. Install all the decoders that are used by pillow.
a. For Ubuntu use the following commands to install the decoders:
apt-get install libjpeg-dev
apt-get install libfreetype6-dev
apt-get install zlib1g-dev
apt-get install libpng12-dev
b. For Centos use the following commands to install the decoders:
yum install zlib zlib-devel
yum install libjpeg libjpeg-devel
yum install freetype freetype-devel

3. Make sure you have a folder named 'DESIGNSEEDS' in your build directory.
All the original images from design seeds are kept in this folder.

4. Make sure that you have two empty folders called slices and palettes in
your build directory. All the image slices will be saved in this folder and
all the test palette will be generated in this folder. These palettes are
created for testing visually from the original palette of a particular image.

5. Make sure that the directory path you are running this code is supplied to
the read_directory(COORD_FILE, PATH) function. Path should be an absolute path
to directory where all the original images are located in the folder name
DESIGNSEEDS.
Absolute path is required and may change if you are using a different OS such
as Windows. Make sure your path is set correctly and all the above dependencies
are satisfied.
'''

import os

from PIL import Image
from PIL import ImageDraw

from Chameleon.nearest_palette import create_distance_matrix

currentimage =""

def whiteness_test(color_tuples, isinitial = True):  # : list) -> bool:
    '''
    This function basically test if the given list of RGB colors pass or fail
    the whiteness test which helps in making the decision of finding the white
    line which will be used for splitting the image and the palette.
    Expects color_tuples to be a list of min / max points that are generated
    by the call (image).getextrema(). Whiteness is considered the threshold
    of an RGB color >= 240.

    isinitial is for finding initial white line that separates image and palette
    initial has higher threshold

    Args:
        color_tuples: A list of color tuples which are extracted through the
                      getextrema() function. These are the RGB color values
                      for either a horizontal or a vertical strip in process.

    Returns:
        A boolean value: Either True or False based on the test.
    '''
    if color_tuples is None:
        return False

    for min_max in color_tuples:
        if isinitial is False:
            if min_max[0] + min_max[1] + min_max[2] < 248*3:
                return False
        if isinitial is True and min_max[0] < 240:
            return False
    return True


def extract_palette(palette_image, orientation, order_by_L = False):
    '''
    This function extracts the palette colors from the already cropped palette
    given as input. Basically, it takes an image of a palette and attempts to
    extract the non-white, major colors within it. Once it has identified all
    major colors from the palette it will return them as a list of RGB tuples.

    Args:
        palette_image: An image which is basically a palette. This input is
                       a palette which is cropped from the original design-seed
                       image.
        orientation: A required argument which decides the orientation of the
                     image - it could be either a horizontal split or vertical
                     split. We extract this information to create the reduced
                     points for the color extraction.

    Returns:
        A list of RGB tuples which have the palette colors extracted from the
        given palette image as an input.
    '''
    if orientation == "horizontal":
        reduce_points = (0, int(palette_image.size[1] / 4),
                         palette_image.size[0],
                         int(palette_image.size[1] / 4) + 5)
    elif orientation == "vertical":
        reduce_points = (int(palette_image.size[0] / 4), 0,
                         int(palette_image.size[0] / 4) + 5,
                         palette_image.size[1])
    else:
        return list()

    palette_line = palette_image.crop(reduce_points)

    c_list = [t for t in palette_line.getcolors(800000)
              if not (t[1][0] >= 240 and t[1][1] >= 240
                      and t[1][2] >= 240) and t[0] > 50]
    #this still results in duplicates
    for t in palette_line.getcolors(800000):
        print(t)
        break
    #remove unneened
    c_list = [ y for x,y in c_list]
    if len(c_list[0]) ==4:
        c_list =[(x,y,z) for x,y,z,f in c_list]
    if order_by_L:
        return sorted(c_list, key = lambda c: c[0]*0.2126 + c[1]*0.7152+ c[2]*0.0722) 
    else:
        return c_list

def extract_palette_keep_order(palette_image, orientation):
    '''
    This function extracts the palette colors from the already cropped palette
    given as input. Basically, it takes an image of a palette and attempts to
    extract the non-white, major colors within it. Once it has identified all
    major colors from the palette it will return them as a list of RGB tuples.

    Args:
        palette_image: An image which is basically a palette. This input is
                       a palette which is cropped from the original design-seed
                       image.
        orientation: A required argument which decides the orientation of the
                     image - it could be either a horizontal split or vertical
                     split. We extract this information to create the reduced
                     points for the color extraction.

    Returns:
        A list of RGB tuples which have the palette colors extracted from the
        given palette image as an input.
    '''
    global currentimage
    if orientation == "horizontal":
        reduce_points = (0, int(palette_image.size[1] / 4),
                         palette_image.size[0],
                         int(palette_image.size[1] / 4) + 1)
    elif orientation == "vertical":
        reduce_points = (int(palette_image.size[0] / 4), 0,
                         int(palette_image.size[0] / 4) + 1,
                         palette_image.size[1])
    else:
        return list()

    palette_line = palette_image.crop(reduce_points)


    if orientation == "vertical":
        #print('vertical: rotating')
        palette_line=palette_line.transpose(Image.ROTATE_90)
    palette_line.save("testing/"+currentimage+str(palette_line.size[0])+".png")
    #traverse line and write color when white is encountered
    #take avg
    #print(palette_line)
    #print('size', palette_line.size)
    c_list =[]
    cur_color =[]
    #if orientation == "vertical":
    # for i in range(0, palette_line.size[1]):
    #     #print(i, p, whiteness_test([(p[0], p[1], p[2])], False), whiteness_test([(p[0], p[1], p[2])], True))
    #     p =palette_line.getpixel((0,i))
    #     #print(p)
    #     if whiteness_test([(p[0], p[1], p[2])], False) is True or i ==palette_line.size[1]-1:
    #         #print('white', len(cur_color))
    #         if len(cur_color)>0:
    #             print('white', len(cur_color))
    #             r=[c[0] for c in cur_color]
    #             g=[c[1] for c in cur_color]
    #             b=[c[2] for c in cur_color]
    #             c_list.append((int(sum(r) / len(r)),int(sum(g) / len(g)), int(sum(b) / len(b)) ))
    #             print(c_list)
    #             cur_color =[]
    #     else:
    #         #print('not white', len(cur_color))
    #         cur_color.append((p[0], p[1], p[2]))
    # # else:
    for i in range(0, palette_line.size[0]):
        p =palette_line.getpixel((i,0))
        #print(i, p, whiteness_test([(p[0], p[1], p[2])], False), whiteness_test([(p[0], p[1], p[2])], True))
        if whiteness_test([(p[0], p[1], p[2])], False) is True or i ==palette_line.size[0]-1:
            #print('white', len(cur_color))
            if len(cur_color)>0:
                #print('white', len(cur_color), cur_color)
                r=[c[0] for c in cur_color]
                g=[c[1] for c in cur_color]
                b=[c[2] for c in cur_color]
                #print(r)
                c_list.append((int(sum(r) / len(r)),int(sum(g) / len(g)), int(sum(b) / len(b)) ))
                #print(c_list)
                cur_color =[]
        else:
            #print('not white', len(cur_color))
            cur_color.append((p[0], p[1], p[2]))
            
    return c_list


def read_vertical_line(image, vbox):  # (image: Image, vbox: tuple) -> tuple:
    '''
    Processes a single vertical line and determines whether that line is
    sufficiently white. If line is not white, exit. Otherwise, extract the
    palette and find the image. Image continues from first non-white line found
    within the same current direction.

    Args:
        image: A design-seed image is passed as an input.
        vbox: A tuple of coordinates point for the vertical thin strip.
              (left of 1 pixel of width, top of image,
                  right of 1 pixel of width, bottom of image)

    Returns:
        A tuple with the Image crop points, Palette crop points and RGB list
        of colors as extracted from the cropped palette. Note if a non-white
        line is found it simply returns False.
    '''
    vline = image.crop(vbox)
    #vline.save("testing/line_ver.png")
    v_color_list = vline.getextrema()
    #print(v_color_list, 'extrema')
    if whiteness_test(v_color_list) is False:
        return False


    image_crop = ()
    palette_crop = (vbox[0] + 1, 0, image.size[0], image.size[1])
    list_of_colors = extract_palette_keep_order(image.crop(palette_crop),
                                     orientation="vertical")
    # list_of_colors = extract_palette(image.crop(palette_crop),
    #                                  orientation="vertical", order_by_L= True)
    if len(list_of_colors) < 3:
        return False

    box_width = vbox[2]
    while box_width > 0:
        vbox = (box_width - 1, 0, box_width, image.size[1])
        vline = image.crop(vbox)
        v_color_list = vline.getextrema()
        if whiteness_test(v_color_list) is True:
            box_width -= 1
        else:
            image_crop = (0, 0, box_width - 1, image.size[1])
            break
    # print(vbox)
    #print(image_crop, palette_crop, list_of_colors)
    return image_crop, palette_crop, list_of_colors


def read_horizontal_line(image, hbox):
    '''
    Processes a single horizontal line and determines whether that line is
    sufficiently white. If line is not white, exit. Otherwise, extract the
    palette and find the image. Image continues from first non-white line found
    within the same current direction.

    Args:
        image: A design-seed image is passed as an input.
        hbox: A tuple of coordinates point for the vertical thin strip.
              (leftmost of image, upper of 1 pixel line,
                  rightmost of image, bottom of 1 pixel line)

    Returns:
        A tuple with the Image crop points, Palette crop points and RGB list
        of colors as extracted from the cropped palette. Note if a non-white
        line is found it simply returns False.
    '''
    global currentimage
    hline = image.crop(hbox)
    #hline.save("testing/line_hor.png")
    h_color_list = hline.getextrema()
    if whiteness_test(h_color_list) is False:
        return False

    image_crop = ()
    palette_crop = (0, hbox[1] + 1, image.size[0], image.size[1])
    list_of_colors = extract_palette_keep_order(image.crop(palette_crop),
                                     orientation="horizontal")
    # list_of_colors = extract_palette(image.crop(palette_crop),
    #                                  orientation="horizontal", order_by_L=True)
    if len(list_of_colors) < 4: # or len(list_of_colors) >8: second case removal of suplacets tajkes care of
        print("Invalid number of colors: ", len(list_of_colors), currentimage)
        return False

    box_height = hbox[3]


    while box_height > 0:
        hbox = (0, box_height - 1, image.size[0], box_height)
        hline = image.crop(hbox)
        h_color_list = hline.getextrema()

        if whiteness_test(h_color_list) is True:
            box_height -= 1
        else:
            image_crop = (0, 0, image.size[0], box_height - 1)
            break
    # print(hbox)
    #print(image_crop, palette_crop, list_of_colors)
    return image_crop, palette_crop, list_of_colors


def process_image(image):
    '''
    This function takes an design-seed image as input and returns a tuple with
    the image's crop points, palette's crop points and the RGB colors which are
    extracted from the palette. It basically processes the images by making a
    thin vertical and horizontal lines; like a raster scan in both vertical and
    horizontal directions. It will run the scan in both the directions until it
    first identifies the white line which satisfies the whiteness test. For all
    cases it will simply increase the thin strip of both the orientations in
    plus one forward direction. For vertical line it will go from right to left
    and for horizontal line it will go bottom to top to identify the white line
    that satisfies the whiteness test. After it has identified the crop points
    for the input image it will simply return it and breaks the scanning loop.
    If however, no crop points are identified it will return a bad tuple.

    Args:
        image: A input image from design-seed for processing.

    Returns:
        A tuple with crop points and RGB colors extracted from the palette or a
        bad tuple if no crop points were identified.
    '''

    # -5 from left, bottom to ignore white lines existing as a border.
    width = image.size[0] - 5
    height = image.size[1] - 5
    # Check the horizontal and vertical lines.
    while width > 0 or height > 0:
        # The result will contain an (image, palette) coordinates in a tuple.
        vertical_box = (width - 1, 0, width, image.size[1])
        horizontal_box = (0, height - 1, image.size[0], height)
        #print( image.size, vertical_box,horizontal_box )
        if width > 0:
            vertical_result = read_vertical_line(image, vertical_box)
            if vertical_result is False:
                width -= 2
            else:
                return vertical_result
        if height > 0:
            horizontal_result = read_horizontal_line(image, horizontal_box)
            if horizontal_result is False:
                height -= 2
            else:
                return horizontal_result
        #break
    # No white line was found
    print("No white line was found")
    return ("bad", "tuple")

def file_len(fname):
    """ get line count"""
    i=0
    #with #open(fname) as f:
    for i, l in enumerate(fname):
        pass
    return i + 1


def remove_duplicates(dir):
    processed_files = []
    removed_files = []
    #read palettes
    color_points = open("backend/cropbox.txt").read()
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    files = [f for f in os.listdir("backend/DESIGNSEEDS/")]
    print("Total number of files:", len(files))
    for filename in files:
        #check for similar filenames
        #check characters just before . (.png)
        name =''
        try:
            if '.' in filename:
                name = filename[0:filename.index('.')]
            if '_' in name:
                name = name[0:name.index('_')]
            #remove trailing numbers
            last_index = len(name)-1
            i = len(name)-1
            while i>1:
                #print(name[i])
                if name[i].isnumeric():
                    #print('remove', name[i])
                    last_index =i
                    i = i-1
                else:
                    break
            name = name[0:last_index]

            #print(filename, name)
            #break
        except:
            print('Problem parsing:', name)

        #find similar named images
        similar = [f for f in files if f not in processed_files and f!=filename and name in f]
        #print(similar)
        if len(similar)>0:
            pal_color_list = get_palette(color_points, filename)
            if len(pal_color_list)>0:
                #compare palette
                for s in similar:
                    #get palette
                    pal_color_list_s = get_palette(color_points, s)
                    #compare
                    # print(pal_color_list, filename)
                    # print(pal_color_list_s, s)
                    #if same number of colors
                    if len(pal_color_list) ==len(pal_color_list_s):
                        #compare colors
                        count_matching =0
                        for c in pal_color_list:
                            if c in pal_color_list_s:
                                count_matching = count_matching+1
                        #definitely duplicate!
                        if count_matching/len(pal_color_list)>0.8:
                            print('Duplicate!!!')
                            #remove from files
                            removed_files.append(s)
                            #delete from Designseeds and slices
                            try:
                                os.remove(cur_dir+"/DESIGNSEEDS/"+s)
                            except:
                                print(s , "does not exist in DESIGNSEEDS")
                            try:
                                os.remove(cur_dir+"/DESIGNSEEDS/slices/"+s)
                            except:
                                print(s , "does not exist in DESIGNSEEDS/slices")
                            
                        else:
                            if count_matching>3:
                                print(filename, s, count_matching, "out of ", len(pal_color_list))
                            processed_files.append(s)
                    else:
                        processed_files.append(s)
                    if s in files:
                        files.remove(s)

        processed_files.append(filename)
    print("Total number of unique files:", len(processed_files))
    print("Total number of duplicate files:", len(removed_files))

import re
def get_palette(color_points, imagename):
    start_index = color_points.find(imagename)
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

    return pal_color_list


def test_palette_extraction(original_images_path, imagename):
    i = Image.open(original_images_path + imagename)
    white_line = process_image(i)
    if white_line != ('bad', 'tuple'):
        # Extract colors from the results of process_image
        # Sometimes elements of rgb_values had 4 instead of 3 values.
        rgb_values = [elem[1][0:3] for elem in white_line[2]]
        print(rgb_values)
        #remove close colors
        rgb_values= remove_duplicate_palette_colors(rgb_values, 20)
        print(rgb_values)
        return  rgb_values

def remove_duplicate_palette_colors(rgb_values, threshold):
    """
    Calculate eucledian distance between colors
    Where distance<threshold, assume it's the same color
    """
    dist_matrix = create_distance_matrix(rgb_values, rgb_values)
    #print(dist_matrix)
    #print(len(rgb_values))
    duplicates =[]
    for j in range(0,len(rgb_values)):
        for k in range(0,len(rgb_values)):
            if k>j and dist_matrix[j][k]<threshold:
                if k not in duplicates:
                    duplicates.append(rgb_values[k])
    #print(duplicates)
    for d in duplicates:
        if d in rgb_values:
            rgb_values.remove(d)
    return rgb_values

from cluster import cluster_image
from Chameleon.website_color_process import LAB_to_RGB
def add_file_manually(textfile_path, original_images_path, image_path, imagename):
    '''
    This function opens a text file and wtries the crop points and RGB colors
    as extracted from the palette by opening each of the design-seed images
    already saved to the disk. It will open an image as specified in the input
    path one by one and send to for processing. After the images processed from
    the process_image() function it will write the contents of it to the file.

    Args:
        textfile: A file pointer object for creating and opening a file to
                  write.
        original_images_path: Absolute path of the location where design-seed
                              images are stored on the local disk.

    Returns:
        None
    '''
    appendfile = open(textfile_path, "a+")
    #check if already exists
    lines = appendfile.readlines()
    exists =0
    for line in lines:
        if imagename in line:
            print(imagename, "already in file!")
            exists =1
            break
    if not exists:
        # Image processing locally
        i = Image.open(image_path + imagename)

        appendfile.write(imagename + ' I:' + str(0) + ' P:'
                           + str(0) + '\n')

        #use clustering to get palette
        lab_values=cluster_image(i, k_clusters=5)
        rgb_values =[]
        for l in lab_values:
            print(l)
            rgb=LAB_to_RGB(l)      
            rgb_values.append(rgb)
        print(rgb_values)   
        appendfile.write(str(rgb_values).strip("[]"))
        appendfile.write('\n')
        i.save(original_images_path + "slices/" + imagename)
        # # Create a visible palette for visualing the retrieved colors
        palette_im = Image.new('RGB', (500, 300))
        draw = ImageDraw.Draw(palette_im)
        for pos, color_val in enumerate(rgb_values):
            draw.rectangle([int(pos * 500 / len(rgb_values)), 0,
                            int(pos * 500 / len(rgb_values) +
                                500 / len(rgb_values)), 300], color_val)
        del draw
        palette_im.save(original_images_path + "palettes/" + imagename)
    appendfile.close()


def read_directory(textfile_path, original_images_path, overwrite):
    '''
    This function opens a text file and wtries the crop points and RGB colors
    as extracted from the palette by opening each of the design-seed images
    already saved to the disk. It will open an image as specified in the input
    path one by one and send to for processing. After the images processed from
    the process_image() function it will write the contents of it to the file.

    Args:
        textfile: A file pointer object for creating and opening a file to
                  write.
        original_images_path: Absolute path of the location where design-seed
                              images are stored on the local disk.

    Returns:
        None
    '''
    # #if exists
    lines =[]
    appendfile=""
    if overwrite or not os.path.exists(textfile_path):
        appendfile = open(textfile_path, "w")

    else:
        textfile = open(textfile_path, "r")
        lines= textfile.readlines()
        textfile.close()
        appendfile = open(textfile_path, "a+")

    num_processed =0
    num_skipped =0
    num_error =0
    print(original_images_path)
    for imagename in os.listdir(original_images_path):
        #print(imagename)
        #have to check both file and folder
        skipflag = False
        skipflag1 = False
        # Skip files that are not .png, .jpeg, nor .jpg
        for line in lines:
            if imagename in line:
                #print(imagename, " already in cropbox")
                skipflag = True
                break
        if imagename in os.listdir(original_images_path+ "slices/"):
            skipflag1 = True
        else:
            print(imagename, "NOT in slices" )
        if (imagename.find(".png") == -1 and
                imagename.find(".jpeg") == -1 and
                imagename.find(".jpg") == -1) or (skipflag and skipflag1):
            #print('Skipping', imagename)
            num_skipped = num_skipped+1
            continue

        white_line=process_one_image(original_images_path,imagename)
        if white_line != ('bad', 'tuple'):
            # Write to the file. Becuase rgb_values is a list, remove the []
            appendfile.write(imagename + ' I:' + str(white_line[0]) + ' P:'
                               + str(white_line[1]) + '\n')
            appendfile.write(str(white_line[2]).strip("[]"))
            appendfile.write('\n')
            #print(str(white_line[2]).strip("[]"))
            num_processed = num_processed+1
        else:
            num_error = num_error+1
    appendfile.close()
    print("Number of Designseed images: ",len(os.listdir(original_images_path)))
    print("Number of processed images: ", num_processed)
    print("Number of skipped images: ", num_skipped)
    print("Number of errors: ", num_error)
    print("Total: ", num_error+num_skipped+num_processed)


def process_one_image(original_images_path,imagename):
    #print('Processing', original_images_path+imagename)
    # Image processing locally
    global currentimage
    currentimage = imagename
    i = Image.open(original_images_path + imagename)
    rgb_values =[]
    white_line = process_image(i)
    # Writing to file for verification or tests.
    if white_line != ('bad', 'tuple'):
        try:
            # Extract colors from the results of process_image
            # Sometimes elements of rgb_values had 4 instead of 3 values.
            # Code ignores any beyond R, G, B
            #rgb_values = [elem[1][0:3] for elem in white_line[2]]
            rgb_values =  white_line[2]
            #remove close colors
            # print(len(rgb_values), 'before')
            rgb_values= remove_duplicate_palette_colors(rgb_values, 20)
            # print(len(rgb_values), 'after')
            # i.crop(white_line[0]).save(
            #     original_images_path + "slices/" + imagename)
            # i.crop(white_line[1]).save(
            #     original_images_path + "palettes/" + imagename)
        except:            
            print('ERROR Saving: ', imagename)

    else:
        print('ERROR CROPPING IMAGE: ', imagename)
    #print(rgb_values)
    return (white_line[0], white_line[1], rgb_values)

def process_one_image_test(original_images_path,imagename):
    result = process_one_image(original_images_path,imagename)
    print(result)

def get_all_actual_palettes(textfile_path, original_images_path, DS=True):
    '''
    This function returns extracted cleaned-up palettes 

    Args:
        textfile - cropbox.txt that already contains extracted palettes, all that we need to do is plot them
        original_images_path: Absolute path of the location where design-seed
                              images are stored on the local disk.

    Returns:
        None
    '''
    textfile = open(textfile_path, "r")
    color_points= textfile.read()
    textfile.close()

    num_processed =0
    num_error =0
    num_skipped =0
    palettes ={}
    #test_list =[]
    #test_list.append("FloraTones8.png")
    #for imagename in test_list:
    for imagename in os.listdir(original_images_path):
        if (imagename.find(".png") == -1 and
                imagename.find(".jpeg") == -1 and
                imagename.find(".jpg") == -1) or ("_" in imagename and imagename.index("_") == len(imagename) -1):
            #print('Skipping', imagename)
            num_skipped = num_skipped+1
            continue
        pal_color_list = []
        #print('Processing', original_images_path+imagename)
        try:
            start_index = color_points.find(imagename)
            #print(start_index)
            start_index = color_points.find('\n', start_index) + 1
            end_index = color_points.find('\n', start_index)
            #print(end_index)
            rgb_color_str = color_points[start_index:end_index]
            #print(rgb_color_str)

            # Convert string into list of RGB tuples
            rgb_patt = re.compile("[(](?P<r>.*?)[,][ ](?P<g>.*?)[,][ ](?P<b>.*?)[)]")
            #print(rgb_color_str)
            for res in rgb_patt.finditer(rgb_color_str):
                pal_color_list.append((int(res.group("r")),
                                       int(res.group("g")),
                                       int(res.group("b"))))
            palettes[imagename] = pal_color_list
        except:
            print('Error getting palette for', imagename)
    return palettes


def render_all_actual_palettes(textfile_path, original_images_path, DS=True):
    '''
    This function renders extracted palettes to /palettes folder

    Args:
        textfile - cropbox.txt that already contains extracted palettes, all that we need to do is plot them
        original_images_path: Absolute path of the location where design-seed
                              images are stored on the local disk.

    Returns:
        None
    '''
    textfile = open(textfile_path, "r")
    color_points= textfile.read()
    textfile.close()

    num_processed =0
    num_error =0
    num_skipped =0
    #test_list =[]
    #test_list.append("FloraTones8.png")
    #for imagename in test_list:
    for imagename in os.listdir(original_images_path):
        if (imagename.find(".png") == -1 and
                imagename.find(".jpeg") == -1 and
                imagename.find(".jpg") == -1) or ("_" in imagename and imagename.index("_") == len(imagename) -1):
            print('Skipping', imagename)
            num_skipped = num_skipped+1
            continue
        pal_color_list = []
        #print('Processing', original_images_path+imagename)
        try:
            start_index = color_points.find(imagename)
            #print(start_index)
            start_index = color_points.find('\n', start_index) + 1
            end_index = color_points.find('\n', start_index)
            #print(end_index)
            rgb_color_str = color_points[start_index:end_index]
            #print(rgb_color_str)

            # Convert string into list of RGB tuples
            rgb_patt = re.compile("[(](?P<r>.*?)[,][ ](?P<g>.*?)[,][ ](?P<b>.*?)[)]")
            #print(rgb_color_str)
            for res in rgb_patt.finditer(rgb_color_str):
                pal_color_list.append((int(res.group("r")),
                                       int(res.group("g")),
                                       int(res.group("b"))))

        except:
            print("Error getting palette")
        #print('here', len(pal_color_list))
        if len(pal_color_list)>0:
            print(pal_color_list)
            if DS==True:
                render_palette(pal_color_list, original_images_path + "palettes/" + imagename+"_DS", 'rgb')  
            else:
                render_palette(pal_color_list, original_images_path + "palettes/" + imagename+"_", 'rgb')   
            num_processed = num_processed+1
        else:
            num_error = num_error+1
            print('ERROR PROCESSING IMAGE: ', imagename)
    print("Processed", num_processed, ", error", num_error, ', skipped', num_skipped)

import shutil
from Chameleon.website_color_process import discover_palette, render_palette, discover_k_best_palettes
def cluster_and_match_image_set(folder, k=1):
    """find closes designseeds image - for experiments
    if k >1 get k best matches and rank them"""
    for imagename in os.listdir(folder):
        try:
            # # Skip files that are not .png, .jpeg, nor .jpg
            #also skip rendered palettes
            if imagename.find("_KMEANS_") >0 or\
            imagename.find("_DS_") >0 or\
                (imagename.find(".png") == -1 and \
               imagename.find(".jpeg") == -1 and \
               imagename.find(".jpg") == -1):
                continue
            #try:
            print("Processing", imagename)
            result =[]
            if k>1:                
                result =discover_k_best_palettes(folder, imagename, k)
            else:
                img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist = discover_palette(folder, imagename, 1)
                result.append((img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist))
            #print('Got nearest palette - rendering', pal_clr_list_lab)
            render_name=folder+imagename+"_DS_"
            #print(render_name)
            for img_cluster_lab, pal_clr_list_lab, result_filename, img_cluster_centr_lab, dist in result:
                print("Matched:", result_filename, dist)
                render_palette(pal_clr_list_lab, render_name+"_DS_"+str(dist), 'LAB')    
                pal_clr_list_lab = sorted(pal_clr_list_lab, key = lambda c: c[0]) 
                render_palette(pal_clr_list_lab, render_name+"_"+str(dist), 'LAB')  
                #copy matched image,too
                shutil.copyfile("backend/DESIGNSEEDS/"+result_filename, folder+imagename+"_"+str(int(dist))+"_"+result_filename) 
            # except:
            #     print("Smth went wrong", imagename)
        except Exception as e: 
            print("Error", imagename, e)


# import click
# @click.command()
# @click.option('--readfile', type=click.STRING, default="",
#               help='Supply an image file to test')
# @click.option('--shout/--no-shout', default=False)

def main():#readfile, shout):
    COORD_FILE_READ = open("backend/cropbox.txt", "r+")
    COORD_FILE_APP = open("backend/cropbox.txt", "a")

    #if readfile != "":
    #    print("You supplied ", readfile)
    #sys.exit()
    # imagename = "AutumnColor.png"
    #test_palette_extraction("backend/DESIGNSEEDS/", "CaminitoColor.png")

    #commented out because takes time and not needed every time
    #remove_duplicates("DESIGNSEEDS/")
    overwrite=1
    COORD_FILE_PATH = "backend/cropbox.txt"
    PATH = r"backend/DESIGNSEEDS/"
    #read_directory(COORD_FILE_PATH, PATH, overwrite)

    #process_one_image_test(PATH,"AlpHues600.png")
    #cluster_and_match_image_set("/home/linka/python/autoimage_flask/testing/art/", 1)
    
    #add_file_manually( "backend/cropbox.txt", r"backend/DESIGNSEEDS/", "/home/linka/python/autoimage_flask/testing/selected_images/", "macke1.png")
    
    #get_all_actual_palettes("backend/cropbox.txt", "backend/DESIGNSEEDS/", DS=True)
    cluster_and_match_image_set("/home/linka/python/autoimage_flask/testing/surveys/images/", 2)


if __name__ == '__main__':
    main()
