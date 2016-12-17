"""
pip3 install colorthief:
A Python module for grabbing the color palette from an image.
"""
import os
from collections import defaultdict

import numpy
import scipy.misc
from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, sRGBColor
from munkres import Munkres

IMAGE_EXTENSIONS = set(['.png', '.jpg', '.jpeg', '.gif'])#, '.svg' behaves like .css

from sklearn.cluster import *
def cluster_image(image, k_clusters=2):
    '''
    Clusters an image into k cluster points. Then, converts each
    color point from RGB to LAB color format.

    Args:
        image: An open PIL image
        k_clusters: Number of clusters to produce. Defaulted to 10.
    Returns:
        cluster_list: A list containing elements in the following format:
                        [(name), (array of colors)]
    Replaced vq with sklearn
    '''
    arr = scipy.misc.fromimage(image)
    shape = arr.shape
    if len(shape) > 2:
        arr = arr.reshape(scipy.product(shape[:2]), shape[2])
        # print(arr)
        # rgblist = [sRGBColor(z[0] / 255, z[1] / 255, z[2] / 255) for z in arr]
        # rgblist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
        # lablist = [convert_color(x, LabColor) for x in rgblist]
        # lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x in lablist])
        # codes, dist = scipy.cluster.vq.kmeans2(lablist, k_clusters, iter=20)
        # print('LEN clusters', len(codes))
        # del dist
        k_means_ex = KMeans(n_clusters=k_clusters)
        x = k_means_ex.fit_predict(arr)
        codes = k_means_ex.cluster_centers_
        #shape should be (k_clusters,3)
        if(codes.shape!=(k_clusters, 3)):
            #print('palette: bad shape', codes.shape)
            codes1= numpy.delete(codes, numpy.s_[-1:], 1)
            #print(codes1)
            return codes1
        else:
            return codes
        end = time.time()
        print('Clustering took',end-start )
        return codes
    else:
        print('Invalid image')
        return []


def find_size_colors(image_name):

    img = Image.open(image_name)
    image_size = (img.size)
    width = image_size[0]
    hieght = image_size[1]
    # img = img.resize((150, 150))
    image_color = img.getcolors(width*hieght)
    print("width ", width, "hieght", hieght)
    print("number of colors", len(image_color))
    return width, hieght

def find_palletes(image_name, count):

    color_thief = ColorThief(image_name)
    dominant_color = color_thief.get_color(quality=1)
    palette = color_thief.get_palette(color_count=count)
    print("palette", palette)
    print("dominant_color", dominant_color)
    return dominant_color, palette


def store_colors_of_pixeles1(image_name):
    """
    In this function we store all distinct colors of pixelsd Ph  into a dictionary
    """
    start = time.time()
    im = Image.open(image_name)
    rgb_im = im.convert('RGB') # for .gif images
    by_color = defaultdict(int)
    # for pixel in im.getdata():
    for pixel in rgb_im.getdata():  # for .gif images
        mylab = RGB2LAB1(pixel)
        by_color[mylab] += 1
    # for c in by_color:
    #     print("color_dictionary", c)
    end = time.time()
    #print('store_colors_of_pixeles1 took', end - start)
    return by_color


def calculate_new_color(palette, colors, distance_L, distance_A, distance_B, param, Phi, count):
    '''
    this function converts all the colors of an image to new_colors
    '''
    w = [1/count for i in range(count)]
    start = time.time()
    # print("distance_L", distance_L)
    # print("distance_A", distance_A)
    # print("distance_B", distance_B)
    new_color = defaultdict(int)
    # new_color = dict()
    t = LabColor(0.0, 0.0, 0.0)
    a = numpy.array(Phi)
    palette_lab = [RGB2LAB1(p) for p in palette]
    for mylab in colors.keys():
        # w = [0 for z in range(count)]
        # print("color in RGB", LAB2RGB1(mylab))
        # print("color in lab space", mylab)
        tempphi = [0 for z in range(count)]
        if (param != 0):
            for index, p in enumerate(palette_lab):
                temp_lab = (mylab.lab_l, mylab.lab_a, mylab.lab_b)
                p1 = (p.lab_l, p.lab_a, p.lab_b)
                r = CalculateLABDistance2(temp_lab, p1)
                tempphi[index] = math.exp(- r * param)

            b = numpy.array(tempphi)
            try:
                w = numpy.linalg.solve(a, b)
            except Exception as e:
                print(str(e))
        # print("weight", w)
        delta_L = 0
        delta_A = 0
        delta_B = 0
        scale = 0
        for weight in w:
            scale = scale + max(weight, 0)
        for index, weight in enumerate(w):
            if(weight > 0):
                # print("weight/scale", weight/scale, " distance_L[index]", distance_L[index])
                delta_L = delta_L + weight/scale * distance_L[index]
                delta_A = delta_A + weight/scale * distance_A[index]
                delta_B = delta_B + weight/scale * distance_B[index]
        t.lab_a = delta_A + mylab.lab_a
        t.lab_b = delta_B + mylab.lab_b
        t.lab_l = mylab.lab_l
        # t.lab_l = delta_L + mylab.lab_l
        # print(t)
        # print("new color in lab space", t)
        rgb = LAB2RGB1(t)
        # print("new color in rgb", rgb)
        LAB = (mylab.lab_l, mylab.lab_a, mylab.lab_b)
        new_color[LAB] = rgb
    #print("size of new_color list", len(new_color))
    # print("key, value", new_color.keys(), new_color.values())
    end = time.time()
    #print('calculate_new_color took', end - start)
    return new_color


def changing_image_colors(im, new_colors):
    # for k in new_colors.keys():
    #     print(k)
    #print('in changing_image_colors')
    start =time.time()
    counter = 0
    Error =''
    #im = Image.open(image_name)
    pixels = im.load()
    width, height = im.size
    for x in range(width):
        for y in range(height):
            try:
                # print("orginal rgb", pixels[x, y], "counter", counter)
                counter = counter + 1
                mylab = RGB2LAB1(pixels[x, y])
                LAB = (mylab.lab_l, mylab.lab_a, mylab.lab_b)
                # print("get", LAB, new_colors[LAB])
                # print("get", LAB, new_colors.get(LAB))
                pixels[x, y] = new_colors.get(LAB)
            except Exception as e:
                #print('Failed',mylab,LAB,e)
                if Error =='':
                    Error = str(e)
                #break
    #im.save(image_name)
    end = time.time()
    #print('changing_image_colors took', end - start)
    if len(Error)>0:
        print('There was an error during recoloring',Error)
    return im

def create_distance_matrix(clusters1, clusters2):
    '''
    Calculates the distance between all points within clusters1
    and clusters2 and stores the result in a distance matrix.

    This function simply creates a distance matrix which uses the code points
    from the random image and the code points from the training set. It applies
    the euclidean distance on the LAB code code points and find the distance
    for each of the k x k elements and builds a matrix, which is simply a list
    of list. Here, k is the parameter k used in while applying the k-means
    algorithm.

    Args:
        clusters1: Code points in LAB format from the training set.
        clusters2: Code points in LAB format computed for random image.

    Returns:
        A list of list which is the distance matrix for each of the code points.
    '''
    distance_matrix = list()
    for color_x in clusters1:
        row = list()
        for color_y in clusters2:
            # print(color_x)
            row.append(euclidean_distance(color_x, color_y))
        distance_matrix.append(row)
    return distance_matrix
def euclidean_distance(point1, point2):
    '''
    Used for calculating the distance between two points. Both
    points must have the same number of dimensions.

    Args:
        point1: A list of points
        point2: Another list of points
    Returns:
        None: if point1 and point2 are not equal length
        result: A single float representative of the euclidan distance
    Raises:
        TypeError: if either parameter is not iterable.

    >>> euclidean_distance([1], [2,3])
    >>> euclidean_distance([1,2], [1,2])
    0.0
    >>> euclidean_distance([1,2], [2,3])
    1.4142135623730951
    >>> euclidean_distance((1,2,4), (9,2,3))
    8.06225774829855
    '''
    if len(point1) != len(point2):
        return None

    distance = 0

    # distance += (x2 - x1)^2 for each x2, x1 in points2, points1
    for value in range(len(point1)):
        distance += (point2[value] - point1[value])**2

    return math.sqrt(distance)

def matching(old_palette, new_palette):
    #print(old_palette)
    # print(new_palette)
    result = create_distance_matrix(old_palette, new_palette)
    mymunk = Munkres()
    bipartite_pairings = mymunk.compute(result)
    #print(bipartite_pairings)
    # potential_distance = calculate_distance(bipartite_pairings, result)
    return bipartite_pairings


def LAB2RGB(t):
    """
    this function does not work correctly
    """
    rgb_temp = convert_color(t, sRGBColor)
    rgb_temp = sRGBColor(rgb_temp.rgb_r, rgb_temp.rgb_g, rgb_temp.rgb_b)
    # print("convert rgb", numpy.uint8(rgb_temp.rgb_r*255), numpy.uint8(rgb_temp.rgb_g*255), numpy.uint8(rgb_temp.rgb_b*255))
    rgb = numpy.uint8(rgb_temp.rgb_r*255), numpy.uint8(rgb_temp.rgb_g*255), numpy.uint8(rgb_temp.rgb_b*255)
    return rgb
def LAB2RGB1(Q):

    var_Y = (Q.lab_l + 16) / 116
    var_X = Q.lab_a / 500 + var_Y
    var_Z = var_Y - Q.lab_b / 200
    if (var_Y > 0.206893034422):
        var_Y = math.pow(var_Y, 3)
    else:
        var_Y = (var_Y - 16 / 116) / 7.787
    if (var_X > 0.206893034422):
        var_X = math.pow(var_X, 3)
    else:
        var_X = (var_X - 16 / 116) / 7.787
    if (var_Z > 0.206893034422):
        var_Z = math.pow(var_Z, 3)
    else:
        var_Z = (var_Z - 16 / 116) / 7.787
    X = 95.047 * var_X
    Y = 100 * var_Y
    Z = 108.883 * var_Z
    var_X = X / 100
    var_Y = Y / 100
    var_Z = Z / 100
    var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
    var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570
    if (var_R > 0.0031308):
        var_R = 1.055 * math.pow(var_R, 1 / 2.4) - 0.055
    else:
        var_R = 12.92 * var_R
    if (var_G > 0.0031308):
        var_G = 1.055 * math.pow(var_G, 1 / 2.4) - 0.055
    else:
        var_G = 12.92 * var_G
    if (var_B > 0.0031308):
        var_B = 1.055 * math.pow(var_B, 1 / 2.4) - 0.055
    else:
        var_B = 12.92 * var_B
    R = int(var_R * 255)
    G = int(var_G * 255)
    B = int(var_B * 255)
    # I added this part because some times they were less than zero
    if(R < 0):
        R = 0
    if(G < 0):
        G = 0
    if(B < 0):
        B = 0
    return (R, G, B)

def RGB2LAB(c):
    if len(c)==3:
        r, g, b = c
    if len(c)==4:
        r, g, b , x= c
    mylab = sRGBColor(r/255, g/255, b/255)
    mylab = convert_color(mylab, LabColor)
    mylab.lab_l = round(mylab.lab_l, 4)
    mylab.lab_a = round(mylab.lab_a, 4)
    mylab.lab_b = round(mylab.lab_b, 4)
    return mylab
def RGB2LAB1(c):
    """
    RGB->XYZ->LAB from easyRGB.com
    """
    if len(c)==3:
        r, g, b = c
    if len(c)==4:
        r, g, b , x= c
    var_R = (r / 255)  # R from 0 to 255
    var_G = (g / 255)  # G from 0 to 255
    var_B = (b / 255)  # B from 0 to 255
    if (var_R > 0.04045):
        var_R = math.pow((var_R + 0.055) / 1.055, 2.4)
    else:
        var_R = var_R / 12.92
    if (var_G > 0.04045):
        var_G = math.pow((var_G + 0.055) / 1.055, 2.4)
    else:
        var_G = var_G / 12.92
    if (var_B > 0.04045):
        var_B = math.pow((var_B + 0.055) / 1.055, 2.4)
    else:
        var_B = var_B / 12.92
    var_R = var_R * 100
    var_G = var_G * 100
    var_B = var_B * 100
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    var_X = X / 95.047
    var_Y = Y / 100
    var_Z = Z / 108.883
    if (var_X > 0.008856):
        var_X = math.pow(var_X, 1/3)
    else:
        var_X = (7.787 * var_X) + (16 / 116)

    if (var_Y > 0.008856):
        var_Y = math.pow(var_Y, 1/3)
    else:
        var_Y = (7.787 * var_Y) + (16 / 116)
    if (var_Z > 0.008856):
        var_Z = math.pow(var_Z, 1/3)
    else:
        var_Z = (7.787 * var_Z) + (16 / 116)
    L = (116 * var_Y) - 16
    A = 500 * (var_X - var_Y)
    B = 200 * (var_Y - var_Z)
    mylab = LabColor(L, A, B)
    return mylab
def CalculateLABDistance(palette1, palette2):
    """
    this function calculates distance two colors in lab space
    """
    my_rgb1 = sRGBColor(palette1[0] / 255, palette1[1] / 255, palette1[2] / 255)
    my_lab1 = convert_color(my_rgb1, LabColor)
    my_rgb2 = sRGBColor(palette2[0] / 255, palette2[1] / 255, palette2[2] / 255)
    my_lab2 = convert_color(my_rgb2, LabColor)
    return euclidean_distance([my_lab1.lab_a, my_lab1.lab_b], [my_lab2.lab_a, my_lab2.lab_b])
def CalculateLABDistance1(palette1, palette2):
    '''
    this function calculates the C' - C (distance between two matched colors)
    I used the Palette-based Photo Recoloring implementation for this function
    '''
    K1, K2 = 0.045, 0.015
    my_lab1 = RGB2LAB1(palette1)
    my_lab2 = RGB2LAB1(palette2)
    l1, a1, b1 = my_lab1.lab_l, my_lab1.lab_a, my_lab1.lab_b
    l2, a2, b2 = my_lab2.lab_l, my_lab2.lab_a, my_lab2.lab_b
    del_L = l1 - l2
    c1 = math.sqrt(a1*a1 + b1*b1)
    c2 = math.sqrt(a2*a2 + b2*b2)
    c_ab = c1 - c2
    h_ab = (a1-a2)*(a1-a2)+(b1-b2)*(b1-b2) - c_ab*c_ab
    return del_L*del_L + c_ab * c_ab / (1+K1*c1)/(1+K1*c1) + h_ab / (1+K2*c1)/(1+K2*c1)
    # return (l1-l2)*(l1-l2)+(a1-a2)*(a1-a2)+(b1-b2)*(b1-b2)
def CalculateLABDistance2(palette1, palette2):
    '''
    This function calculate the C' - C (when C' is in labspace and C in RGB)
    '''
    K1, K2 = 0.045, 0.015
    # my_lab2 = RGB2LAB1(palette2)
    l1, a1, b1 = palette1[0], palette1[1], palette1[2]
    # l2, a2, b2 = my_lab2.lab_l, my_lab2.lab_a, my_lab2.lab_b
    l2, a2, b2 = palette2[0], palette2[1], palette2[2]
    del_L = l1 - l2
    c1 = math.sqrt(a1*a1 + b1*b1)
    c2 = math.sqrt(a2*a2 + b2*b2)
    c_ab = c1 - c2
    h_ab = (a1-a2)*(a1-a2)+(b1-b2)*(b1-b2) - c_ab*c_ab
    return del_L*del_L + c_ab * c_ab / (1+K1*c1)/(1+K1*c1) + h_ab / (1+K2*c1)/(1+K2*c1)
    # return (l1-l2)*(l1-l2)+(a1-a2)*(a1-a2)+(b1-b2)*(b1-b2)

def Calculatediff(color1, color2):
    my_lab1 = RGB2LAB1(color1)
    my_lab2 = RGB2LAB1(color2)
    l1, a1, b1 = my_lab1.lab_l, my_lab1.lab_a, my_lab1.lab_b
    l2, a2, b2 = my_lab2.lab_l, my_lab2.lab_a, my_lab2.lab_b
    del_L = l2 - l1
    del_A = a2 - a1
    del_B = b2 - b1
    # print("here")
    return del_L, del_A, del_B
def calculate_param(palette):
    '''
   Color Transfer using variant RBF & GMM.
   param: mean distance between all pairs of colors in the original palette
   RBF_param_coff: influction parameter in gaussian function of RBF.
   I used the Palette-based Photo Recoloring implementation for this function
    '''
    RBF_param_coff = 5
    tot, totr = 0, 0
    if(len(palette) == 1):
        print('len(palette) == 1 ?')
        return 0
    else:
        for u in range(len(palette)):
            for v in range(u+1, len(palette)):
                r = CalculateLABDistance1(palette[u], palette[v])
                tot = tot + 1
                totr = totr + math.sqrt(r)
        totr = totr / tot
        if(totr == 0):
            return 0
        else:
            return RBF_param_coff / (totr*totr)

def calculate_Phi(palette, param, count):
    '''
    this function calculate phi for colors of a palette
    implementation is matched with Palette-based Photo Recoloring code
    '''
    Phi = list()
    for index1, p1 in enumerate(palette):
        row = [0 for z in range(count)]
        for index2, p2 in enumerate(palette):
            r = CalculateLABDistance1(p1, p2)
            # row.append(math.exp(- r * param))
            row[index2] = math.exp(- r * param)
        Phi.append(row)
    return Phi

def recolor(img, palette1, count, colors):
    start = time.time()
    palette = cluster_image(img, count)
    # print(len(palette), palette)
    # print(len(palette1), palette1)
    # print( palette1.shape)
    pair = matching(palette, palette1)
    #print(pair[0])
    color_distL = list()
    color_distA = list()
    color_distB = list()
    for p in pair:
        # print(palette[p[0]], palette1[p[1]])
        diffL, diffA, diffB = Calculatediff(palette[p[0]], palette1[p[1]])
        color_distL.append(diffL)
        color_distA.append(diffA)
        color_distB.append(diffB)
        # print(color_distL, color_distA, color_distB)
    # color_distL, color_distA, color_distB = [Calculatediff(palette[p[0]], palette1[p[1]]) for p in pair]
    # print(color_distL)
    param = calculate_param(palette)
    #print("param", param)
    #print('param', param, 'into phi')
    Phi = calculate_Phi(palette, param, count)
    #print('done calculate_Phi', Phi)
    # print("Phi", Phi)
    #colors = store_colors_of_pixeles1(path+"/image/"+image_name)
    # for c in colors:
    #     print(LAB2RGB(c))
    #print("before this function calculate_new_color")
    # new_colors = defaultdict(int)
    new_colors = calculate_new_color(palette, colors, color_distL, color_distA, color_distB, param, Phi, count)
    palette1 = [(int(a), int(b), int(c)) for a, b, c in palette1]
    #print('palette1',palette1)
    # print(new_colors)
    # for p in pair:
    #     print(p[0], p[1])
    #     mylab = RGB2LAB(palette[p[0]])
    #     LAB = (mylab.lab_l, mylab.lab_a, mylab.lab_b)
    #     new_colors[LAB] = palette1[p[1]]
    #     # pixels[x, y] = new_colors.get(LAB)
    #     # for i, inde in enumerate(colors):
    #     #     if(inde.lab_l - mylab.lab_l == 0 and inde.lab_a - mylab.lab_a == 0 and inde.lab_b - mylab.lab_b == 0):
    #     #         print("I found it", i)
    #     #         index_main = i
    #     #         new_colors[index_main] = palette1[p[1]]
    # print(new_colors)
    end = time.time()
    #print('recolor  took',end-start )
    return new_colors


def test_recoloring():
    path = os.getcwd()
    path = os.path.join(path,'code')
    image_name_to_recolor = "faclogo4.png"
    palette_image = "DS_palette_.png"
    print(path+"/"+palette_image)
    palette_img = Image.open(path+"/"+palette_image)
    palette1 = cluster_image(palette_img, 5)
    count = len(palette1)  # number of color in a palette
    #print(count, palette1)
    print(path+"/recolorable/"+image_name_to_recolor)
    recolor_one_image(palette1,target_palette_lab,  count, os.path.join(path,"recolorable",image_name_to_recolor), os.path.join(path,image_name_to_recolor))

import time
def test_recolor_all_recolorable():
    """
    for every image in recolorable folder
    recolor
    write recolored back to parent folder
    """
    start = time.time()
    path = "/home/linka/python/autoimage_flask/uploads/tmpqwabhwee"
    print(path)
    palette_image = "DS_palette.png"
    palette_img = Image.open(os.path.join(path,palette_image))
    palette1 = cluster_image(palette_img, 5)
    recolor_all_recolorable(os.path.join(path, "recolorable"), palette1, palette1, (153, 0, 51), (153, 0, 51))
    end = time.time()
    print('Total recoloring took', end-start)


from os.path import isfile,join
from os import listdir
def recolor_all_recolorable(path, target_palette, target_palette_lab, old_bg, new_bg):
    """
    for every image in recolorable folder
    recolor
    write recolored back to parent fold
    target_palette has to be RGB with no weights!!!!
    """
    count = len(target_palette)  # number of color in a palette
    #print('target_palette',target_palette)
    if os.path.exists(path):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for f in  onlyfiles:
            try:
                recolor_one_image(target_palette, target_palette_lab, count, join(path, f), join(path, f).replace("recolorable/", ""), old_bg, new_bg)
            except Exception as e:
                #at least copy the original to dest folder
                shutil.copy( join(path, f), join(path, f).replace("recolorable/", ""))  
                print("Problem recoloring", f,":  ", e)


def background_detect(img_path, color):
    delta = 10  # for difference between passed color and border colors
    thresh = 0.5  # for checking the color frequency
    img = Image.open(img_path)
    rgb_img = img.convert('RGB')
    w, h = img.size
    # print(w, h)
    # extract border of the image
    first_row = [rgb_img.getpixel((0, i)) for i in range(h)]
    last_row = [rgb_img.getpixel((w-1, i)) for i in range(h)]
    first_coloumn = [rgb_img.getpixel((i, 0)) for i in range(w)]
    last_coloumn = [rgb_img.getpixel((i, h-1)) for i in range(w)]
    # print(last_coloumn)
    colors_list = first_row + last_row + first_coloumn + last_coloumn
    # print(len(colors_list))
    sum = 0
    for col in colors_list:
        if(abs(color[0]-col[0]) < delta and abs(color[1]-col[1]) < delta and abs(color[2]-col[2]) < delta):
            sum += 1
            # print("here")
    freq = sum / len(colors_list)
    #print(freq)
    if(freq >= thresh):
        return 1
    else:
        return 0


import math
from Chameleon.weighted_pallete import cluster_big_set
def recolor_one_image(target_palette,  target_palette_lab, count,  img_path, img_save_path, old_bg, new_bg):
    start = time.time()
    img = Image.open(img_path)
    #try:
    img_temp = img
    # # if img.mode == "RGBA" or "transparency" in img.info:
    # #     img.load()
    # #     background = Image.new("RGB", img.size, (255, 255, 255))
    # #     #print(img.split())
    # #     background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    # #     background.save(img_path)
    # #     img = Image.open(img_path)
    # if img.mode in ('RGBA', 'LA'):
    #     img.load()
    #     background = Image.new("RGBA", img.size, (255, 255, 255))
    #     background.paste(img, mask=img.split()[3]) # 3 is the alpha channel
    #     background.save(img_path)
    #     img = Image.open(img_path)
    # elif (img.mode == 'P' and 'transparency' in img.info):
    #     alpha = img.convert('RGBA').split()[-1]
    #     img.putalpha(alpha)
    #     img.save(img_path)
    #     img = Image.open(img_path)
    is_transparent_bg =  0
    transparency = None
    #print('transparency',img.mode , img.info)
    if (img.mode == 'P' and 'transparency' in img.info) or img.mode in ('RGBA', 'LA'):
        alpha = img.convert('RGBA').split()[-1]  # creat and save alpha channel
        is_transparent_bg =  1
        if 'transparency' in img.info:
            transparency = img.info['transparency']
    img = img.convert('RGBA')
    colors = store_colors_of_pixeles1(img_path)
    #need to check how many distinct colors - if less that in palette, cluster the palette
    num_colors = img.getcolors()
    if (num_colors!=None and len(num_colors)<count) or (num_colors==None and len(colors)<count):
        numcolors =len(colors)
        if num_colors!=None:
            numcolors = len(num_colors)
        target_palette_clustered_weighted= cluster_big_set(target_palette_lab, numcolors, None)
        #convert to RGB
        target_palette=[]
        for l1,l2,l3,w in target_palette_clustered_weighted:
            lab =LabColor(l1,l2,l3)
            r=LAB2RGB1(lab) 
            target_palette.append(r)#(r.rgb_r, r.rgb_g, r.rgb_b))
        count =len(target_palette)
    #here the problem happens
    new_colors = recolor(img, target_palette, count, colors)
    delta =10
    if(not is_transparent_bg and background_detect(img_path, old_bg)):
        #print('has bg')
        for mylab in colors.keys():
            # r, g, b = LAB2RGB1(mylab)
            # if(abs(r-old_bg[0]) < delta and abs(g-old_bg[1]) < delta and abs(b-old_bg[2]) < delta):
            temp = RGB2LAB1(old_bg)
            #if(abs(mylab.lab_l-temp.lab_l) < delta and abs(mylab.lab_a-temp.lab_a) < delta and abs(mylab.lab_b-temp.lab_b) < delta):
            if math.sqrt(math.pow(mylab.lab_l-temp.lab_l,2) + math.pow(mylab.lab_a-temp.lab_a, 2) +math.pow(mylab.lab_b-temp.lab_b,2)) < delta:
                LAB = (mylab.lab_l, mylab.lab_a, mylab.lab_b)
                # print(new_colors.get(LAB), "before")
                new_colors[LAB] = new_bg
                # print(new_colors.get(LAB), "after")
    im = changing_image_colors(img, new_colors)
    #if (img_temp.mode == 'P' and 'transparency' in img_temp.info) or img_temp.mode in ('RGBA', 'LA'):

    if is_transparent_bg:
        im.putalpha(alpha)  # reapply alpha chanel after recoloring
        # if transparency!=None:
        #     im.info['transparency'] = transparency
        #     im = im.convert('P')
        #print('transparency', img.mode , im.mode, im.info)
    save_image(im, img_path, img_save_path)
    end = time.time()
    print('Recoloring ', img_path, ' took', end-start )

import shutil
def save_image(im, img_path, img_save_path):
    #take care of extensions that end with smth else
    fileName,fileExtension = os.path.splitext(img_path)
    old_img_save_path = img_save_path
    if fileExtension not in IMAGE_EXTENSIONS:
        #have to be careful to not overwrite
        #print(img_save_path)
        img_save_path = img_save_path.replace(fileExtension, "")
        img_save_path = img_save_path.replace("/code", "")
        #print(img_save_path)
        im.save(img_save_path)
        #move from img_save_path to old_img_save_path
        shutil.move( img_save_path, old_img_save_path)  
    else:
        im.save(img_save_path)
    #print(img_save_path, old_img_save_path)

from cluster import  deserialize
def test_recolor_one_image(DS_imgname, img_name, path):
    
    img_path=os.path.join(path,img_name )
    img_save_path=os.path.join(path,"_"+img_name )
    clusterfilename= "backend/clusterlistbytes.txt"
    training_clusters = deserialize(clusterfilename)
    colors=[p for n, p in training_clusters if n == DS_imgname]
    target_palette_lab=[(p[0], p[1], p[2]) for p in list(colors[0])]# (n, p) for n, p in training_clusters if n == DS_imgname]
    target_palette= []
    for lab in target_palette_lab:
        lab =LabColor(lab[0], lab[1], lab[2])
        r=LAB2RGB1(lab) 
        target_palette.append(r)
    # target_palette = []
    # target_palette.append((204, 102, 0))
    # target_palette.append((255, 0, 0))
    # target_palette.append((73, 143, 218))
    # target_palette.append((153, 0, 51))    
    # target_palette.append((114, 193, 209))
    # # target_palette.append((38, 86, 123))
    # # target_palette.append((214, 221, 224))
    # # target_palette.append((73, 143, 218))
    # # target_palette.append((24, 19, 67))    
    # # target_palette.append((114, 193, 209))
    # target_palette_lab = []
    # target_palette_lab.append((44.036102815179319, -7.8309862040425617, -31.138117358281423, 15))
    # target_palette_lab.append((38.399389290982455, 7.2343700373902129, -45.323687747367408, 18))    
    # target_palette_lab.append((17.32021668253989, 18.440498698621404, -39.039036882965028, 37))    
    # target_palette_lab.append((63.731322912936449, -19.340691703640815, -21.229149423090337, 11))
    count = len(target_palette)
    print(target_palette) #[(38, 86, 123), (214, 221, 224), (73, 143, 218), (24, 19, 67), (114, 193, 209)]
    print(target_palette_lab)
    old_bg=(249, 249, 249) 
    new_bg=(96, 164, 242)
    recolor_one_image(target_palette, target_palette_lab, count,  img_path, img_save_path, old_bg, new_bg)

if __name__ == '__main__':
    path = "/home/linka/python/autoimage_flask/uploads/test/"
    img_name = "nomadic_matt_logo.png"#creepfest.gif"
    DS_imgname = "FloraBright_6.png"
    test_recolor_one_image(DS_imgname, img_name, path)
    # mylab=RGB2LAB1((255,255,255))
    # print(mylab)
    #test_convert_color()
    #test_recoloring()
    #test_recolor_all_recolorable()