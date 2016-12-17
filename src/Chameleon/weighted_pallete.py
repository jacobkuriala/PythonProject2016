'''
Created on 2/22/2016

@author: Polina

This file weights designseeds palette based on color distribution in the image

Get designseeds palette colors
Cluster image into same number of colors as palette
Predict cluter for palette colors
Assign weights to palette colors
Render weighted palette
'''
import os
import pickle
from collections import Counter

from PIL import Image
from colormath.color_conversions import convert_color
from colormath.color_objects import LabColor, AdobeRGBColor
from image_palette_process import get_all_actual_palettes
from numpy import bincount
from website_color_process import render_palette_vertical, render_matching, render_palette_vertical_weights

from cluster import cluster_image_predict


def get_weighted_palettes():
    
    #get_designseed_palettes
    #RGB
    DSpalettes = get_all_actual_palettes("backend/cropbox.txt", "backend/DESIGNSEEDS/")
    weighted_pallettes =[]
    for filename, DS in DSpalettes.items():
        path =''
        k = len(DS)
        if k>0:
            try:
                print(filename, ',  DS:', DS, len(DS))
                adobergbpalette = [AdobeRGBColor(r, g, b, True) for r, g, b in DS]
                labpal = [convert_color(x, LabColor) for x in adobergbpalette]
                arraylabpal = [[x.lab_l, x.lab_a, x.lab_b] for x in labpal]
                #get orig corresponding image
                im = Image.open(os.path.join("backend/DESIGNSEEDS/",'slices', filename))
                im = im.convert('P', palette=Image.ADAPTIVE).convert("RGB")
                #print(len(arraylabpal))
                #cluster
                labpalette, labels, predict_labels = cluster_image_predict(im, arraylabpal, k_clusters=k)
                #print('labpalette',labpalette)
                #print('labels',labels.shape)
                #print('predict_labels',predict_labels)
                labpalette_withweights =[] # weight is % rounded to int. like 50 (%)
                weights = bincount(labels)
                #print('weights', weights)
                total = sum(weights)
                #print('total', total)
                i=0
                for lab in labpalette:
                    labpalette_withweights.append((lab, int(weights[i]*100/total)))
                    #print(lab, int(weights[i]*100/total))
                    i =i+1

                #print(labpalette_withweights)
                #render both
                path = os.path.join('backend','DESIGNSEEDS', 'weighted_palettes')
                #render_palette_vertical(labpalette, os.path.join(path, filename + "_cluster"), "LAB")
                sorted_labpalette_withweights = sorted(labpalette_withweights, key=lambda d: d[1]) 
                render_palette_vertical_weights(sorted_labpalette_withweights,   os.path.join(path,filename + "_weighted_cluster"), "LAB", 500)
                
                #check cluster prediction
                matching =[]
                for i in range(0,k): #cluster #s
                    for j in range(0,k): #lab palette indexes
                        if predict_labels[j] ==i: #cluster numbers match
                             matching.append((labpalette[i], arraylabpal[j]))

                render_matching(matching, os.path.join(path, filename + "_clustermatching"), 'LAB')
                #match to DS palette - assume the weight of that cluster
                #some palette colors get assigned to same cluster - need to break
                #assign conflicts to closest unassigned
                ds_weighted =[]
                #assigned_clusters = set(predict_labels)
                unassigned_clusters = set(p for p in labels if p not in predict_labels)
                assigned_clusters = list(Counter(predict_labels).keys() )
                duplicate_clusters = list(Counter(predict_labels).values())
                duplicates =[]
                unassigned =[]
                # print('assigned_clusters', assigned_clusters)
                # print('unassigned_clusters', unassigned_clusters)
                # print('duplicate_clusters', duplicate_clusters)
                for i in range(0, k):#cluster #s
                    found =0
                    for j in range(0,k): #lab palette indexes
                        if predict_labels[j] ==i: #cluster numbers match
                            found =1
                            if i in assigned_clusters:
                                count = duplicate_clusters[assigned_clusters.index(i)]
                                #print('count', count)
                                if count>1:#Counter(predict_labels)[i]>1:
                                    #print('duplicate', i,  [d for d,c,w in duplicates])
                                    if i not in [d for d,c,w in duplicates]:
                                        duplicates.append([i, count, labpalette_withweights[i][0]])
                                        #unassigned.append((i, arraylabpal[j]))
                                else:
                                    #print('not duplicate', i)
                                    ds_weighted.append((arraylabpal[j], labpalette_withweights[i][1]))
                
                for j in range(0,k): #lab palette indexes
                    if arraylabpal[j] not in [c for c,w in ds_weighted]:
                        unassigned.append(arraylabpal[j])
                #print(ds_weighted)
                # print('duplicates',duplicates)
                # print('unassigned', unassigned)

                #handle duplicates and unassigned
                for col2 in unassigned:        
                    #for each duplicate there's an unassigned
                    #move duplicates to unassigned
                    #check which duplicate is closest to j's unassigned cluster
                    mindist = 100000
                    closest_cluster =-1
                    closest_col =()
                    closest_cnt =0
                    closest_k =-1
                    for k in range(0, len(duplicates)):
                        i, cnt, col = duplicates[k]
                        if cnt>0:
                            distance = euclidean_distance(col, col2)
                            if distance<mindist:
                                closest_cluster = i
                                closest_col = col
                                closest_cnt =cnt
                                closest_k =k
                                mindist = distance
                    if closest_cluster>-1:
                        ds_weighted.append((col2, labpalette_withweights[closest_cluster][1]))
                        duplicates[closest_k][1] = closest_cnt -1
                        #print('updated', duplicates[closest_k] , closest_cnt -1)
                        # print('removing', [closest_cluster, closest_cnt, closest_col])
                        # duplicates.remove([closest_cluster, closest_cnt, closest_col])
                    else:
                        print("-------------- ERROR  NO CLOSEST CLUSTER???----------------")
                #scale to make 100%?
                total = sum([w for c,w in ds_weighted])
                #print('total for reweight', total)
                #sort by weight
                sorted_ds_weighted = sorted([(c, int(w*100/total)) for c,w in ds_weighted], key=lambda d: d[1]) 
                #print('ds_weighted', sorted_ds_weighted, sum([w for c,w in sorted_ds_weighted]))
                weighted_pallettes.append((filename, sorted_ds_weighted))
                #render_palette_vertical(DS, os.path.join(path,filename+"_DS"), "rgb")
                render_palette_vertical_weights(sorted_ds_weighted,   os.path.join(path,filename+"_weighted_DS"), "LAB", 500)
                if(len(ds_weighted)!=len(DS)):
                    print('/*********************************/')
                    print('weighted palette not same length!!!!!', len(ds_weighted),len(DS))
                    print('/*********************************/')
                #break
            except:
                print('Error processing', filename)
    #write to cropbox_weighted
    write_cropbox_weighted(weighted_pallettes)
    return weighted_pallettes

    
def write_cropbox_weighted(weighted_palettes):
    filename ="backend/weighted_palettes"
    palfile = open(filename, "wb")
    #pickle.dump(cluster_list, cluster_bytes)
    pickle.dump(weighted_palettes, palfile, protocol=2)
    palfile.close()

def read_weighted_palettes():
    filename ="backend/weighted_palettes"
    palfile = open(filename, "rb")
    #pickle.dump(cluster_list, cluster_bytes)
    weighted_palettes=pickle.load(palfile)
    palfile.close()
    print(len(weighted_palettes))
    return weighted_palettes

def weighted_matching():
    """ match user image to a palette taking weights into account
    simply add weight as one more dimension
    weighted matching may produce very different results than color matching
    give user an option which one to use 
    """
    # userimage
    # allDSclusters
    #    result = create_distance_matrix(test_codes, colors)
    #    mymunk = Munkres()
    #    bipartite_pairings = mymunk.compute(result)
    #    # print(filename)
    #    potential_distance = calculate_distance(bipartite_pairings, result)
    #    #print('potential_distance, current_distance', potential_distance, current_distance)
    #    return potential_distance
    pass

def get_weights_from_clustering(centroids, test_labels):
    weightedresult=[]
    weights = bincount(test_labels)
    total = sum(weights)
    i=0
    for lab in centroids:
        weightedresult.append([lab[0], lab[1], lab[2], int(weights[i]*100/total)])
        i =i+1
    return weightedresult

from cluster import cluster_image, deserialize
from Chameleon.matching import match_and_calculate_distance
def find_nearest_cluster_weighted(starting_filename, dest_path):#, mindist=0, prevmatches=[]):
    '''
    Takes the filename of an image, opens the image, deserializes the
    list of lists which contains a filename and its corresponding clusters,
    and clusters the test image.

    For each cluster in the training set, calculates the distance between
    the images by using Euclidean Distance on the nearest cluster color
    points between the two images and summing them all together.

    The result is the image in the training set that has the smallest
    distance from the test image.

    Args:
        starting_filename: An input random image.

    Returns:
        Closest image match from the training set.
    '''
    print('In find_K_nearest_clusters_weighted')
    starting_img = Image.open(starting_filename)
    starting_img = starting_img.resize((150, 150))
    centroids, test_labels  = cluster_image(starting_img)
    weightedresult=get_weights_from_clustering(centroids, test_labels)
    print('Clustered test image')
    clusterfilename= "backend/clusterlistbytesweighted.txt"
    training_clusters = deserialize(clusterfilename)
    #print(training_clusters)
    #print('got DS clusters')
    current_image_clusters, current_image = get_nearest_cluster_weighted(training_clusters, weightedresult)
    render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in weightedresult],   os.path.join(dest_path, "user_pal_w"), "LAB", 500)
    render_palette_vertical(centroids,   os.path.join(dest_path, "user_pal"), "LAB")
    render_palette_vertical([(c1, c2, c3) for c1, c2, c3, w in current_image_clusters],   os.path.join(dest_path, "DS_palette_no_w"), "LAB")
    render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in current_image_clusters],   os.path.join(dest_path, "DS_palette"), "LAB", 500)
    #print(current_image_clusters)
    return current_image, current_image_clusters, centroids

def get_nearest_cluster_weighted(training_clusters, weightedresult):
    current_distance = -1
    mindist=0
    current_image = None
    current_image_clusters = []
    #print('Looking for match...')
    for filename, colors in training_clusters:
        # Get training image
        # For each training image, create a matrix of euclidian
        potential_distance = match_and_calculate_distance(weightedresult, colors)
        #print(filename, potential_distance)
        #print('potential_distance, current_distance', potential_distance, current_distance)
        if (current_distance <0) or (potential_distance < current_distance and current_distance>mindist):
            current_distance = potential_distance
            current_image = filename
            current_image_clusters = colors
    print('Min distance:', current_distance, current_image)
    return current_image_clusters, current_image

def clamp(x): 
  return max(0, min(x, 255))

import numpy
from Chameleon.website_color_process import render_palette
from cluster import cluster_array
def matching_algorithm_new_weighted(PAL_color_list_LAB, used_css_rgb_weighted, L_only, results_folder, dest_folder, render_all=False):
    """
    PAL_color_list_LAB is the list of palette colors LAB
    [[85.2787927101103, 1.1833082428296948, 4.302894776701693],
    unique_css_HEX is HEX colors from .css
    results_folder is for rendering

    LAB is only needed for finding closest image
    Use HSL to manipulate brightness
    """
    # print('/****************************************/')
    # print('PAL_color_list_LAB', PAL_color_list_LAB[0])
    # print('used_css_rgb_weighted', used_css_rgb_weighted[0])
    # #print(len(unique_css_HEX), len(used_css_rgb_weighted))
    # print('/****************************************/')
    num_clusters = 0
    done = 0
    big_set = []
    small_set = []
    #big_set_clustered =[]
    big_set_clustered_weighted =[]
    css_is_smallest = 0
    palette_is_smallest = 0
    if render_all:
        # render DS palette
        render_palette_vertical(PAL_color_list_LAB, os.path.join(results_folder,'DS_palette_'), 'LAB')
    #check which set is bigger and cluster bigger set colors 
    #into num_clusters clusters,
    #where num_clusters is the number of colors in the smaller set
    if len(PAL_color_list_LAB) > len(used_css_rgb_weighted):
        print('len(PAL_color_list_LAB) > len(unique_css_HEX)')
        num_clusters = len(used_css_rgb_weighted)
        css_is_smallest = 1
    elif len(PAL_color_list_LAB) < len(used_css_rgb_weighted):
        print('len(PAL_color_list_LAB) < len(unique_css_HEX)')
        num_clusters = len(PAL_color_list_LAB)
        palette_is_smallest = 1
    else:
        done = 1
    if not done:
        print('num_clusters, palette_is_smallest, css_is_smallest', num_clusters, palette_is_smallest, css_is_smallest)
        small_set, big_set, ORIG_CSSRGB_TO_LAB=prepare_colors_for_matching(used_css_rgb_weighted,PAL_color_list_LAB, palette_is_smallest, css_is_smallest)
        if not palette_is_smallest and not css_is_smallest:
            print('WTF')
        big_set_clustered_weighted= cluster_big_set(big_set, num_clusters, results_folder)
    else:
        print('same length - just match')
        big_set_clustered_weighted = big_set
        print('big_set_clustered_weighted',big_set_clustered_weighted[0])
        render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in big_set_clustered_weighted],  results_folder+'/big_set_clustered', "LAB", 500)

    #print('Actual Matching Procedure')
    #print('ORIG_CSSRGB_TO_LAB', ORIG_CSSRGB_TO_LAB)
    # print('small_set', small_set[0])
    # print('big_set_clustered_weighted', big_set_clustered_weighted[0])

    #for testing bipartite_match_css_to_palette
    print('writing bipartite', results_folder)
    test_bytes = open(os.path.join(results_folder, 'bipartite_test_data'), "wb")
    pickle.dump((small_set, big_set_clustered_weighted, palette_is_smallest, css_is_smallest, big_set), test_bytes, protocol=2)
    
    matching, lab_big_set_clustered, lab_small_set = bipartite_match_css_to_palette(small_set, big_set_clustered_weighted, palette_is_smallest, css_is_smallest, results_folder, 1)
    # print('matching', matching)
    # print('lab_big_set_clustered', lab_big_set_clustered)
    # print('lab_small_set', lab_small_set)
    matching_with_centroid ={}
    if palette_is_smallest:
        matching, extended_palette, matching_with_centroid=create_additional_colors(lab_big_set_clustered, big_set, lab_small_set, matching, L_only, results_folder)
    else:
        extended_palette= matching
    if render_all:
        render_matching([(a,c) for a,b,c in sorted(extended_palette, key=lambda x: x[1])], results_folder+'/extended_palette', 'LAB')
    #print('extended_palette', extended_palette)
    if css_is_smallest:
        print('no additional colorsA vector v belongs to cluster i if it is closer to centroid i than any other centroids')

    #print('matching', len(matching))
    #final matching: orig hex(css) to new hex (palette)
    #add #
    #print('ORIG_CSSRGB_TO_LAB', ORIG_CSSRGB_TO_LAB)
    #labtorgb ={}
    final_matching ={}
    for new, orig in matching.items():
        #print( 'new',new,'orig',orig)#, orig[0],orig[1],orig[2])

        ORIG_CSSRGB_TO_LAB_helper = [(k.lab_l, k.lab_a, k.lab_b) for k in ORIG_CSSRGB_TO_LAB.keys()]
        #wtf gets rounded in conversion!!!
        #origlab = LabColor(orig[0],orig[1],orig[2])
        orcol=()
        if type(orig) ==LabColor: #smaller css
            orcol =(new.lab_l, new.lab_a, new.lab_b)
            new=orig
        else:
            orcol = (orig[0],orig[1],orig[2]) 
        rgb = convert_color(new, AdobeRGBColor)
        rgb = rgb.get_upscaled_value_tuple()
        #enforce bounds
        rgb = (clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))
        #labtorgb[new] = rgb
        # rgb = LAB2RGB1(new)
        if orcol in ORIG_CSSRGB_TO_LAB_helper:
            #print('origlab in ORIG_CSSRGB_TO_LAB', orcol)
            for k, v in ORIG_CSSRGB_TO_LAB.items():
                if (k.lab_l, k.lab_a, k.lab_b)==orcol:
                    origrgb= v.get_upscaled_value_tuple()
                    #print('rgb', rgb)       
                    #new_hex =rgb.get_rgb_hex() #'#%02x%02x%02x' % (rgb['rgb_r'],rgb['rgb_g'],rgb['rgb_b'])
                    #new_hex ='#%02x%02x%02x' % (rgb[0]*255,rgb[1]*255,rgb[2]*255)
                    if len(final_matching)>0 and origrgb in final_matching.keys():
                        print('WTH already exists in final_matching!')
                    final_matching[origrgb]=rgb #new_hex
                    #print('final_matching[origrgb]=rgb', origrgb,rgb)
        else:
            print('origlab not in ORIG_CSSRGB_TO_LAB', orcol)
    #print(len(final_matching))
    if(len(final_matching)!=len(big_set) and palette_is_smallest ==1):
        print('Final matching has wrong number of colors!', len(final_matching), len(big_set))
    if(len(final_matching)!=len(used_css_rgb_weighted) and palette_is_smallest ==0):
        print('Final matching has wrong number of colors!', len(final_matching), len(used_css_rgb_weighted))
    #render_all=1
    if render_all:
        render_final_matching(L_only, final_matching, results_folder)
    #print('labtorgb', labtorgb)
    #render_matching(labtorgb, results_folder+'/labtorgb', 'LABRGB')
    return final_matching, matching_with_centroid

from Chameleon.website_color_process import render_separate_palettes
def render_final_matching(L_only, final_matching, results_folder):
    render_matching(final_matching, results_folder+'/final_matching', 'rgb')
    if L_only==1:
        print('L_only==1')
        render_matching(final_matching, results_folder+'/final_matching_Lonly', 'rgb')
        render_separate_palettes( final_matching, results_folder, 'rgb')
        # render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[0]), results_folder+'/final_matching_Lonly', 'hexrgb')
        # render_matching(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder+'/final_matching_Lonly1', 'hexrgb')

        #render_separate_palettes(sorted([(k, v) for k, v in final_matching.items()], key=lambda x:x[1]), results_folder, 'hexrgb')
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

from cluster import euclidean_distance
from Chameleon.website_color_process import check_LAB_boundaries, match_brightness_HSL
def create_additional_colors(lab_big_set_clustered, big_set, lab_small_set, matching, L_only, results_folder):
    print('Creating additional colors')
    #write for testing
    # test_bytes = open(os.path.join(results_folder, 'test_data'), "wb")
    # pickle.dump((lab_big_set_clustered, big_set, lab_small_set, matching), test_bytes, protocol=2)
    
    #this is matching of representative css to palette - doesn't work in this case
    #need to find actual original css hex that matched to palette
    #get all colors in cluster, and find closest to representative

    #find color closest to centroid and map it to palette color

    closest_color_centroid ={}
    #for rendering
    matching_with_centroid ={}
    for centroid in lab_big_set_clustered:
        mindist =float('inf')
        closest_color =0
        for color in big_set:
            #print([centroid[0], centroid[1], centroid[2]], color)
            dist =euclidean_distance([centroid.lab_l, centroid.lab_a, centroid.lab_b], [color[0], color[1], color[2]])# [color.lab_l, color.lab_a, color.lab_b] )
            if mindist>dist:
                mindist=dist
                closest_color = color
        #calc vector between closest color and centroid - to be applied to all other colors in cluster
        #vector = numpy.array([0, 0, 0])
        # we don't want same vector for all colors!!!!
        vector = numpy.subtract(numpy.array([closest_color[0], closest_color[1], closest_color[2]]),  numpy.array([centroid.lab_l, centroid.lab_a, centroid.lab_b]))
        #print(' vector',vector, numpy.array([closest_color.lab_l, closest_color.lab_a, closest_color.lab_b]),  numpy.array([centroid.lab_l, centroid.lab_a, centroid.lab_b]))
        closest_color_centroid[lab_big_set_clustered.index((centroid))] = (closest_color, vector)
        

    #print('closest_color_centroid', closest_color_centroid)
    for k, v in matching.items():
        matching[k]= closest_color_centroid[lab_big_set_clustered.index(v)][0]
        matching_with_centroid[k]= (closest_color_centroid[lab_big_set_clustered.index(v)][0],lab_big_set_clustered.index(v))
    #print('matching_with_centroid', matching_with_centroid)
    #print('matching', len(matching), matching)
    #print('centroid', [v for k,v in closest_color_centroid.items()])
    #print('big set', big_set)
    #render_matching(matching, results_folder+'/matching_clustered', 'LAB')
    #match_for_render= dict((k[0],(v[0], v[1], v[2])) for k,v in matching.items())
    render_matching({k:v for k,v in matching.items()}, results_folder+'/matching_clustered', 'LAB')

    #match L to css even for original palette colors
    if L_only==2:
        print('L_only==2')
        for orig, new in matching.items():
            #print(orig, new)
            orig.lab_l =  match_brightness_HSL(LabColor(new[0],new[1],new[2] ), orig)
            #print('after HSL LAB', orig)
            #print('orig.lab_l, new.lab_l', orig.lab_l, new.lab_l)
        render_matching(matching, results_folder+'/matching_clustered_L', 'LAB')
        print('rendered to', results_folder+'/matching_clustered_L')
    cluster_mapping=[]
    for color in big_set:
        if color not in [v[0] for k,v in closest_color_centroid.items()]:
            #A vector v belongs to cluster i if it is closer to centroid i than any other centroids
            mindist =float('inf')
            closest_centroid =[]
            for centroid in lab_big_set_clustered:
                dist =euclidean_distance([centroid.lab_l, centroid.lab_a, centroid.lab_b], [color[0], color[1], color[2]])
                if mindist>dist:
                    mindist=dist
                    closest_centroid = centroid
            cluster_mapping.append((lab_big_set_clustered.index(closest_centroid), color))

        # else:
        #     print('color in [v[0] for k,v in closest_color_centroid.items()]')
    # crentroids they obviously map to themselves
    render_matching([(lab_big_set_clustered[i], c) for i, c in sorted(cluster_mapping,key=lambda x: x[0])], results_folder+'/cluster_mapping', 'LAB')
    print('rendering to', results_folder+'/cluster_mapping')

    # print('cluster_mapping', cluster_mapping)
    # print(len(matching), len(cluster_mapping))
    # For every other color C[i]: i<>j create a new color by applying a vector c[j]->C[i] to smaller color. 
    #print(matching)
    extended_palette =[]
    for centroid, unmapped_color in cluster_mapping:
        #print('unmapped_color', unmapped_color)
        big_set_color, vector = closest_color_centroid[centroid]
        #add diff between current color and closest color to centroid
        additionalvector = numpy.subtract(numpy.array([unmapped_color[0], unmapped_color[1], unmapped_color[2]]), numpy.array([big_set_color[0], big_set_color[1], big_set_color[2]]))
        pal_color = [s for s,b in matching.items() if b==big_set_color]
        pal_color = pal_color[0]
        #print('big_set_color, pal_color', big_set_color, pal_color)
        #print(' vector',vector)
        if L_only==1:
            new_color=numpy.array( [pal_color.lab_l, pal_color.lab_a, pal_color.lab_b])
            #print(new_color, pal_color.lab_l, numpy.add(vector, additionalvector)[0])
            new_color[0]= pal_color.lab_l+ numpy.add(vector, additionalvector)[0]
            #print('new_color', new_color)
        elif L_only==2:
            #print('L_only==2:')
            new_color_lab = pal_color
            #print(unmapped_color, new_color_lab)
            unmapped_color_lab = LabColor(unmapped_color[0],unmapped_color[1],unmapped_color[2] )
            new_color_lab.lab_l = match_brightness_HSL(unmapped_color_lab, new_color_lab)
            new_color=numpy.array( [new_color_lab.lab_l, new_color_lab.lab_a, new_color_lab.lab_b])
            #render_matching([(unmapped_color, new_color_lab)], 'test_final_'+str(unmapped_color.lab_l), 'LAB')
            #new_color[0]= unmapped_color.lab_l #match L don't change

        else:
            #print('else', L_only)
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
        LabNewColor = LabColor(new_color[0], new_color[1], new_color[2])
        matching[LabNewColor] = unmapped_color#[unmapped_color[0], unmapped_color[1], unmapped_color[2]]
        matching_with_centroid[LabNewColor]= (unmapped_color,centroid)
    #print('extended matching', len(matching), matching)
    #print('matching_with_centroid', matching_with_centroid)
    return matching, extended_palette, matching_with_centroid

import sys
def test_create_additional_colors(results_folder):
    testfile = open(os.path.join(results_folder, 'test_data'), "rb")

    try:
        lab_big_set_clustered, big_set, lab_small_set, matching = pickle.load(testfile)
        print(len(lab_big_set_clustered),len(big_set),len(lab_small_set),len(matching))
        try:
            #L_only =1 # match L leave hue, for new only, leave palette colors as they are
            #L_only =2 # match L even for palette colors
            L_only =0 # shift all L,A,B
            matching, extended_palette, matching_with_centroid=create_additional_colors(lab_big_set_clustered, big_set, lab_small_set, matching, L_only, results_folder)
            render_matching([(a,c) for a,b,c in sorted(extended_palette, key=lambda x: x[1])], results_folder+'/extended_palette_'+str(L_only), 'LAB')
            #sort first by centroid, then by lightness
            final_matching_list = [(k,v[1]) for k,v, in matching_with_centroid.items()]
            final_matching_list_sorted = [(k,v) for k,v in sorted(final_matching_list, key=lambda x:(x[1], x[0].lab_l))]
            render_palette([k for k,v in final_matching_list_sorted],   os.path.join(results_folder, "out_pal_cl"), "LAB")
            final_matching_list = [(v[0],v[1]) for k,v, in matching_with_centroid.items()]
            final_matching_list_sorted = [(k,v) for k,v in sorted(final_matching_list, key=lambda x:(x[1], x[0][0]))]
            render_palette([k for k,v in final_matching_list_sorted],   os.path.join(results_folder, "original_pal_cl"), "LAB")
        except:
            e = sys.exc_info()[0]
            print('Error creating colors! %s'% e)
            raise

    except:
        print('Error reading file or file is empty!')
        raise

def prepare_colors_for_matching(used_css_rgb_weighted,PAL_color_list_LAB, palette_is_smallest, css_is_smallest):
    #print('convert HEX to LAB')
    temp_CSS_LAB = []
    #UNIQUE_CSS_RGB = []
    ORIG_CSSRGB_TO_LAB = {}
    for h,w in used_css_rgb_weighted:
        #get RBG and convert
        # rgb =hex_to_rgb(h)
        # #UNIQUE_CSS_RGB.append(rgb)
        RGB = AdobeRGBColor(h[0]/ 255, h[1]/ 255, h[2]/ 255)
        lab = convert_color(RGB, LabColor)
        temp_CSS_LAB.append((lab, w))
        ORIG_CSSRGB_TO_LAB[lab] = RGB
    UNIQUE_CSS_LAB=  [[x.lab_l, x.lab_a, x.lab_b, w] for x, w in temp_CSS_LAB]
    # print('/****************************************/')
    # print('LEN UNIQUE_CSS_LAB', len(UNIQUE_CSS_LAB), UNIQUE_CSS_LAB[0])
    # print('/****************************************/')
    # #render_palette(UNIQUE_CSS_LAB, results_folder+'/orig_css_conv_to_LAB', 'LAB')
    # #render_palette(UNIQUE_CSS_RGB, 'orig_css_conv_to_rgb', 'rgb')
    # #print('UNIQUE_CSS_LAB', UNIQUE_CSS_LAB)
    lablist =[]
    if palette_is_smallest:
        #print('palette smallest', UNIQUE_CSS_LAB)
        big_set = UNIQUE_CSS_LAB
        small_set = PAL_color_list_LAB
        #lablist = numpy.array([[x.lab_l, x.lab_a, x.lab_b] for x, w in big_set])
    if css_is_smallest:
        #print('css smallest', PAL_color_list_LAB)
        big_set = PAL_color_list_LAB
        small_set = UNIQUE_CSS_LAB
    return small_set, big_set, ORIG_CSSRGB_TO_LAB

def cluster_big_set(big_set, num_clusters, results_folder):
    lablist = numpy.array([[x,y,z] for x,y,z, w in big_set])
    #print('lablist', lablist[0])
    #print('PAL_color_list_LAB', PAL_color_list_LAB)
    #print('clustering bigger set into ', num_clusters)

    #big_set_clustered contains k colors representing k clusters.
    ###todo: how do we know which color in which cluster???
    #big_set_clustered, dist = scipy.cluster.vq.kmeans(lablist, num_clusters)
    centroids, labels=cluster_array(num_clusters, lablist)
    #add weights to clusters
    big_set_clustered_weighted=get_weights_from_clustering(centroids, labels)
    #print('big_set_clustered_weighted',big_set_clustered_weighted[0])
    if results_folder!=None:
        render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in big_set_clustered_weighted],  results_folder+'/big_set_clustered', "LAB", 500)
    return big_set_clustered_weighted

from Chameleon.nearest_palette import create_distance_matrix
from munkres import Munkres
def bipartite_match_css_to_palette(small_set, big_set_clustered_weighted, palette_is_smallest, css_is_smallest, results_folder, _lambda=1):
    matching ={} #palette LAB : css LAB 
    #Match clusters to the colors in the smaller list.
    #find matching using Hungarian algorithm
    #print('small_set',small_set[0])

    #after testing, I came to conclusion that matching color information makes no sense
    #what matters are lightness and proportions
    print(small_set[0])
    print(big_set_clustered_weighted[0])
    #give L more weight -no that does not make sense
    #only use a lambda for W
    small_set_LandW = [(l,a,b,_lambda*w) for l,a,b,w in small_set]
    big_set_LandW = [(l,a,b,_lambda*w) for l,a,b,w in big_set_clustered_weighted]
    print(small_set_LandW[0])
    print(big_set_LandW[0])    
    result = create_distance_matrix(small_set_LandW, big_set_LandW)
    #result = create_distance_matrix(small_set, big_set_clustered_weighted)
    mymunk = Munkres()
    bipartite_pairings = mymunk.compute(result)
    #print('bipartite_pairings', bipartite_pairings)
    lab_big_set_clustered =[]
    lab_small_set=[]
    for b in big_set_clustered_weighted:
        lab = LabColor(b[0], b[1], b[2])
        lab_big_set_clustered.append(lab)
    lab_small_set=[]
    for b in small_set:
        lab = LabColor(b[0], b[1], b[2])
        lab_small_set.append(lab)
    #print('lab_big_set_clustered', lab_big_set_clustered[0])
    #print('lab_small_set', lab_small_set[0])
    for s,b in bipartite_pairings:
        #print(b,s)
        if palette_is_smallest:
            #print('small_set[s]', lab_small_set[s],lab_big_set_clustered[b])
            # print('big_set_clustered[b]', big_set_clustered[b])
            matching[lab_small_set[s]]= lab_big_set_clustered[b]#lab_big_set_clustered[b]          
            #matching.append(small_set[s], lab_big_set_clustered[b])
        if css_is_smallest:
            #print('small_set[b]', lab_small_set[b],lab_big_set_clustered[s])
            matching[lab_small_set[b]]=lab_big_set_clustered[s]            
            #matching.append(small_set[b],lab_big_set_clustered[c])
    #For each Cluster i, C[i], find a color c[j] in C[i] that is closest to j by L value. 
    #print('matching', len(matching))
    render_matching(matching, results_folder+'/DS_css_matching', 'LAB')
    return matching, lab_big_set_clustered, lab_small_set


def test_bipartite_match_css_to_palette():
    #read data
    results_folder='/home/linka/python/autoimage_flask/uploads/test7/'
    data_file = open(os.path.join(results_folder, 'bipartite_test_data'), "rb")
    try:
        result = pickle.load(data_file)
        small_set, big_set_clustered_weighted, palette_is_smallest, css_is_smallest, big_set = result
        _lambda =5
        matching, lab_big_set_clustered, lab_small_set=bipartite_match_css_to_palette(small_set, big_set_clustered_weighted, palette_is_smallest, css_is_smallest, results_folder, _lambda)

        closest_color_centroid ={}
        #for rendering
        matching_with_centroid ={}
        for centroid in lab_big_set_clustered:
            mindist =float('inf')
            closest_color =0
            for color in big_set:
                #print([centroid[0], centroid[1], centroid[2]], color)
                dist =euclidean_distance([centroid.lab_l, centroid.lab_a, centroid.lab_b], [color[0], color[1], color[2]])# [color.lab_l, color.lab_a, color.lab_b] )
                if mindist>dist:
                    mindist=dist
                    closest_color = color
            vector = numpy.subtract(numpy.array([closest_color[0], closest_color[1], closest_color[2]]),  numpy.array([centroid.lab_l, centroid.lab_a, centroid.lab_b]))
            closest_color_centroid[lab_big_set_clustered.index((centroid))] = (closest_color, vector)
            
        for k, v in matching.items():
            matching[k]= closest_color_centroid[lab_big_set_clustered.index(v)][0]
            matching_with_centroid[k]= (closest_color_centroid[lab_big_set_clustered.index(v)][0],lab_big_set_clustered.index(v))
        render_matching({k:v for k,v in matching.items()}, results_folder+'/matching_clustered_'+str(_lambda), 'LAB')

    except Exception as e:
        print(str(e))



from Chameleon.get_palette import download_and_extract_palette
def test_weighted_matching():
    url ='www.rei.com'#'http://www.cs.fsu.edu/department/faculty/sudhir/'# 
    path = 'bg/code'
    used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, old_bg, css_bg=download_and_extract_palette(url, path, '')
    #used_css_rgb_values , unique_css_HEX, used_css_rgb_weighted= get_css_palette_from_css_and_screenshot(path, 20, 0)
    starting_filename = "/home/linka/Desktop/auroreboreale_by_tohad-d8k9co6.jpg"
    current_image,  weightedresult, rgb_centroids=find_nearest_cluster_weighted(starting_filename, '')
    print('current_image', current_image, weightedresult)
    render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in weightedresult],  'code/DS', "LAB", 500)

    L_only=1
    results_folder='code'
    final_matching, matching_with_centroid=matching_algorithm_new_weighted(weightedresult,used_css_rgb_weighted, L_only, results_folder, '')
    #write final matching
    #match L for bg color-convert to LAB
    old_bg_lab = LabColor(old_bg[0],old_bg[1],old_bg[2])
    new_bg_lab = LabColor(final_matching[old_bg][0],final_matching[old_bg][1],final_matching[old_bg][2])
    new_bg_lab.lab_l= old_bg_lab.lab_l
    new_bg_rgb = convert_color(new_bg_lab, AdobeRGBColor) 
    print(old_bg, final_matching[old_bg], new_bg_rgb)
    final_matching[old_bg] = (new_bg_rgb.rgb_r, new_bg_rgb.rgb_g, new_bg_rgb.rgb_b)
    final_matching_bytes = open(os.path.join(results_folder, 'final_matching'), "wb")
    pickle.dump(final_matching, final_matching_bytes, protocol=2)

from Chameleon.website_color_process import find_css #recolor_css_new
import shutil
def test_recolor_css( results_folder):
    #final_matching here is rgb to rgb old:new
    final_matching ={}
    final_matching_file = open(os.path.join(results_folder, 'final_matching'), "rb")
    try:
        final_matching = pickle.load(final_matching_file)
    except:
        print('Error reading file or file is empty!')
    print('final_matching', len(final_matching))
    #css_names don't include html names!
    css_names, html_names = find_css(results_folder, 1) #1 is to save a copy

    print(results_folder, css_names, html_names)
    replaced=recolor_css(final_matching, css_names+html_names)
    print('replaced', replaced)

def recolor_css(matching, css_names):
    '''
    This function swaps the css colors with the generated palatte colors.
    This will replace and write new colors to the css files.

    Args:
        matching - rgb to rgb, orig : new
        css_names - css files names to write to
    '''
    replaced_by_css ={}
    # Open original file and open a file that will contain the swapped colors
    #index.html could have inline styles
    #css_names.append(os.path.join(os.path.dirname(css_names[0]),'index.html'))
    for css in css_names:
        orig_css = os.path.join(os.path.dirname(css),'_'+os.path.basename(css))                    
        shutil.copy2(css, orig_css)  
        #print(css, orig_css)
        replaced=replace_in_css(orig_css, css, matching)
        replaced_by_css[css] =replaced 
        #print('replaced',len(replaced), replaced)
    return replaced_by_css

def test_replace_in_css(results_folder):
    final_matching ={}
    final_matching_file = open(os.path.join(results_folder, 'final_matching'), "rb")
    try:
        final_matching = pickle.load(final_matching_file)
        print(final_matching)
    except:
        print('Error reading file or file is empty!')
    print('final_matching', len(final_matching))
    css = os.path.join(results_folder, 'test.css')
    orig_css = os.path.join(results_folder, '_test.css' ) 
    # css = os.path.join(results_folder, 'sphinx_rtd_theme.css')
    # orig_css = os.path.join(results_folder, '_sphinx_rtd_theme.css' )  
    replaced=replace_in_css(orig_css, css, final_matching)
    #print('replaced', replaced)

def convert_matching_to_hex_rgba(matching):
    matching_=[]
    for origrgb, rgb in matching.items():
        origrgba = 'rgba(%d,%d,%d' % (origrgb[0],origrgb[1],origrgb[2]) #rgba(119,28,50,0.9)
        newrgba = 'rgba(%d,%d,%d' % (rgb[0],rgb[1],rgb[2])
        #print(origrgba, newrgba)
        #this doesn't check bounds and results in invalid hexes
        orig = '#%02x%02x%02x' % (origrgb[0],origrgb[1],origrgb[2])
        new = '#%02x%02x%02x' % (rgb[0],rgb[1],rgb[2])
        # orig = rgb2hex(origrgb[0],origrgb[1],origrgb[2])
        # new = rgb2hex(rgb[0],rgb[1],rgb[2])
        # Letters made uppercase to avoid capitalization misses
        #try 3-letter hexes too!
        if orig[0]=='#':
            orig3 = '#'+orig[1]+orig[3]+orig[5]
        else:
            orig3 =  '#'+orig[0]+orig[2]+orig[4]
        if len(new)>7:
            print('Bad hex!!! orig3, orig, new, origrgba, newrgba', orig3, orig, new, origrgba, newrgba)
        matching_.append((origrgb, rgb, orig3, orig, new, origrgba, newrgba))
    return matching_


import re, codecs
def replace_in_css(orig_css, css, matching):
    #print(orig_css)
    filetext=""
    try:
        file_p = codecs.open(orig_css, 'r', encoding="utf-8")# encoding='latin-1')
        filetext = file_p.readlines()#open(os.path.join(orig_dir,os.path.basename(css))).readlines()
    except:
        file_p = codecs.open(orig_css, 'r', encoding='latin-1')
        filetext = file_p.readlines()#open(os.path.join(orig_dir,os.path.basename(css))).readlines()
    file_p = open(css, "w")
    replaced =set()
    _rgbstring = re.compile(r'#[a-fA-F0-9]{6}$')
    #print(_rgbstring)
    #make sure each # or rgba is in it's own line or there will be a problem
    lines1 = []
    lines = []
    for fline in filetext:
        fline1 = fline.replace('#','\n#')
        lines1=lines1+re.split('\n', fline1)
    for line1 in lines1:
        fline1 = line1.replace('rgba(','\nrgba(')
        lines=lines+re.split('\n', fline1)

    #preprocessing palettes
    matching_ =convert_matching_to_hex_rgba(matching)
    #print('Starting replacement')
    for line in lines:
        #print(line)

        # if is_ascii(line)==0:
        #     print(0, line)
        # if "rgba" in line:
        #     print('start', line, 'end')
        elem_found = False
        for origrgb, rgb, orig3, orig, new, origrgba, newrgba in matching_:
            if orig.upper() in line.upper():
                # Process element
                pattern = re.compile(orig, re.IGNORECASE)
                newline= pattern.sub(new, line)
                #newline =line.replace(orig, new)
                file_p.write(newline)
                # print('__________________________')
                # print('replaced:', line, newline)
                # print('replaced 6 char hex', orig, new)
                elem_found = True
                replaced.add(orig)
                break
            elif orig3.upper() in line.upper():
                #careful here: might match first 3 letters of hex!
                #check if a string is a valid 6-digit hex
                #if so, it's not a match, skip
                is6hex = False
                try:
                    fullhex = line[line.index(orig3):line.index(orig3)+7]
                    #print('fullhex', fullhex)
                    is6hex=bool(_rgbstring.match(fullhex))
                    #print('is6hex', is6hex)
                except:
                    pass#print('couldn''t get substring' ,line)
                if not is6hex:
                    pattern = re.compile(orig3, re.IGNORECASE)
                    newline= pattern.sub(new, line)
                    #newline =line.replace(orig3, new)
                    file_p.write(newline)
                    elem_found = True
                    replaced.add(orig3)
                    # print('__________________________')
                    # print('replaced:',line, newline)
                    # print('replaced 3 char hex', orig, orig3, new)
                    break
            #for testin
            # elif "rgba" in line.lower():
            #     print(line)
            #     print(origrgba, origrgba.lower() in line.lower())
            elif origrgba.lower() in line.lower():
                # Process element
                # pattern = re.compile(origrgba, re.IGNORECASE)
                # newline= pattern.sub(newrgba, line)
                line = line.lower()
                newline =line.replace(origrgba.lower(), newrgba.lower())
                file_p.write(newline)
                elem_found = True
                replaced.add(origrgba)
                # print('__________________________')
                # print('replaced:',line, newline)
                # print('replaced rgba', origrgba, newrgba)
                break
                #print('replaced', orig3, new)
            #break
        if elem_found==False:
            file_p.write(line)
            #print('orig:', line)
    file_p.close()
    return replaced

def evaluate_weight_lambda():
    path = "lambda"
    clusterfilename= "backend/clusterlistbytesweighted.txt"
    training_clusters = deserialize(clusterfilename)

    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in  onlyfiles:
        try:
            #create a subdir for each image
            print(str(f).split('.'))
            imgdir = os.path.join(os.getcwd(),path, f.split('.')[0])
            print('imgdir', imgdir)
            try:
                os.mkdir(imgdir)
            except:
                print('dir exists')

            #cluster and extract palette
            imgpath = os.path.join(os.getcwd(),path,f)
            print('imgpath', imgpath)
            dest =os.path.join(imgdir,f)
            shutil.copy2(os.path.join(path,f), dest)  

            starting_img = Image.open(imgpath)
            starting_img = starting_img.resize((150, 150))
            centroids, test_labels  = cluster_image(starting_img)
            weightedresult=get_weights_from_clustering(centroids, test_labels)
            render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in weightedresult], os.path.join(imgdir, "weighted_input"), "LAB", 500)
            #print('weightedresult', weightedresult)
            #vary lamdba
            _lambda = 0.01
            while _lambda<100:
                lambda_weightedresult =[[w[0], w[1],w[2],_lambda*w[3]] for w in weightedresult]
                #print('lambda_weightedresult', lambda_weightedresult)
                #find closest
                colors=[]
                current_image=''
                colors, current_image = get_nearest_cluster_weighted(training_clusters, lambda_weightedresult)
                #print('colors',colors)
                render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in colors],   os.path.join(imgdir,current_image+'_pal'), "LAB", 500)
                shutil.copy2(os.path.join('backend', 'DESIGNSEEDS', current_image), imgdir)  
                _lambda = _lambda*2
        except:
            print("Problem processing", f)

def test_wo_lambda():
    path = "lambda"
    onlyfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for f in  onlyfiles:
        try:
            imgpath = os.path.join(os.getcwd(),path,f)
            compare_weighted_unweighted_matching(imgpath)
            #break
        except:
            print("Problem processing", f)


from Chameleon.nearest_palette import find_nearest_cluster
def compare_weighted_unweighted_matching(starting_filename):
    #starting_filename = "/home/linka/Desktop/theriver_2_by_seb_m-d9f7lm0.jpg"
    #starting_filename = "/home/linka/Desktop/far_away_by_eintoern-d9ech9d.jpg"
    #starting_filename = "/home/linka/Desktop/g_family_33_by_livingwild-d9hy581.jpg"
    #starting_filename = "/home/linka/Desktop/auroreboreale_by_tohad-d8k9co6.jpg"
    current_image, test_codes, current_image_clusters, current_distance = find_nearest_cluster(starting_filename)
    print('unweighted', current_image, current_distance)
    wcurrent_image, weightedresult, wcurrent_image_clusters, wcurrent_distance=find_nearest_cluster_weighted(starting_filename)
    print('weighted', wcurrent_image,wcurrent_distance)


import time
from Chameleon.get_palette import take_screenshot_selenium
from Chameleon.image_recolor import recolor_all_recolorable
def do_weighted_matching_full_path(url, web_archive, image_name, tmpfolder):#(url,tmpfolder, path, image_name):
    #(loginsource,userid, username,web_url, web_archive, image_name, tmpfolder, path):
    #redis.set(currentkey, "Started")
    start = time.time()
    print(url, web_archive, image_name, tmpfolder)
    #1: create tmp folder in static
    tmpfolderusername = os.path.basename(os.path.normpath(tmpfolder))
    ##logging.info('{} {}'.format('tmpfolderusername', tmpfolderusername))
    #create dir for image or make sure that dir for image exists

    #if I'm manually testing, remove backend from path
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    if os.path.basename(os.path.normpath(path)) =='backend':
        path = os.path.dirname(os.path.normpath(path))
    print(path)

    webpath = os.path.join(tmpfolder, 'code')
    print(webpath)
    recol_path=os.path.join(webpath, "recolorable")

    d = os.path.join(path,"static","out",str(tmpfolderusername))
    if not os.path.exists(d):
        os.makedirs(d)
        print(d)
        ##logging.info('{} {}'.format('made dir', d))
    #2: copy user image to static
    #copy both original and resulting images into  static
    userimagepath =os.path.join(path, 'uploads', tmpfolderusername, image_name)
    dest =os.path.join(d, image_name)
    # print('copy', userimagepath , ' to ', dest)
    # ##logging.info('copy', userimagepath , ' to ', dest)
    shutil.copy2(userimagepath, dest)  

    # #3: download website, take orig screenshot, identify all images, remove all videos
    # classify images into recolorable/nonrecolorable,  move nonrecolorable, 
    # take screenshot w/o recolorable, wo images, return css and image palettes
    now3 = time.time()
    used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, old_bg, css_bg=download_and_extract_palette_wrapper(url, webpath, d)
    now4 = time.time()
    print('TIME FOR download_and_extract_palette', now4 -now3)

    # #4: extract userimage palette and find closest patelle
    current_image,  current_image_clusters, pal_clr_list_lab=find_nearest_cluster_weighted_wrapper(path, userimagepath, d)
    now = time.time()
    print('TIME FOR find_nearest_cluster_weighted', now4 -now)

    #5 calculate new palette for css and recolor css
    L_only=1
    final_matching, new_bg, matching_with_centroid=matching_algorithm_new_weighted_wrapper(pal_clr_list_lab,used_css_rgb_weighted, L_only, webpath, d, old_bg, css_bg)
    final_matching_bytes = open(os.path.join(d, 'final_matching'), "wb")
    pickle.dump(final_matching, final_matching_bytes, protocol=2)
    print('after matching_algorithm_new_weighted_wrapper')
    now1 = time.time()
    print('TIME FOR css palette calculating', now1 -now)
    #print('old_bg, new_bg', old_bg, new_bg)

    #6 recolor files
    recolor_css_new_wrapper(final_matching, css_names, webpath)
    now2 = time.time()
    print('TIME FOR RECOLOR CSS', now2 -now1)


    #7: recolor images
    istart = time.time()
    recolor_all_recolorable_wrapper(recol_path, pal_clr_list_lab, old_bg, new_bg)
    iend = time.time()
    print('Total image recoloring took', iend-istart)

    #8: take final screenshot and render original and resulting palettes to static/out
    # #logging.info('recolored')
    render_results(webpath, final_matching, pal_clr_list_lab, d, html_name, matching_with_centroid)

    # #logging.info('{} {} {}'.format('rerendered recolored',html_path,new_rerender_image_name))
    end = time.time()
    print('TOTAL TIME', end- start)
    return 'Success'

##############################################################################
#do_weighted_matching_full_path pieces
def download_and_extract_palette_wrapper(url, webpath, d):
    used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, old_bg, css_bg=download_and_extract_palette(url, webpath, d)
    #sorted_css = sorted([c for c,w in used_css_rgb_weighted], key=hsl)
    #render_palette_horizontal_weights(used_css_rgb_weighted, os.path.join(d, "original_pal"), "rgb",500)
    render_palette([c for c,w in  sorted(used_css_rgb_weighted, key=lambda x: x[0])], os.path.join(d, "original_pal1"), "rgb")
    #[(a,c) for a,b,c in sorted(extended_palette, key=lambda x: x[1])], results_folder+'/extended_palette', 'LAB')
    #render_palette([c for c,w in used_css_rgb_weighted], os.path.join(d, "original_pal"), "rgb")
    #render_palette(recolorable_image_palette,  os.path.join(d, "original_pal_im"), "rgb")
    return used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, old_bg, css_bg

from Chameleon.website_color_process import get_palette
def find_nearest_cluster_weighted_wrapper(path, userimagepath, d):
    current_image,  current_image_clusters, rgb_centroids=find_nearest_cluster_weighted(userimagepath, d)
    #copy matched image to out folder
    DSsource =os.path.join(path, 'backend', 'DESIGNSEEDS', current_image)
    dest =os.path.join(d, current_image)
    #logging.info('{} {} {} {}'.format('copy', loginsource , ' to ', dest))
    shutil.copy2(DSsource, dest)  
    #get actual DS palette
    color_points = open(os.path.join(path,"backend/cropbox.txt")).read()
    pal_clr_list_lab_no_weight=get_palette(color_points, current_image)
    #map to current_image_clusters to get weights 
    #TODO: this should be done during preporcessing
    #print('current_image_clusters', current_image_clusters)
    current_image_colors= [(c1, c2, c3) for c1, c2, c3,w in current_image_clusters]
    result = create_distance_matrix(current_image_colors, pal_clr_list_lab_no_weight)
    mymunk = Munkres()
    bipartite_pairings = mymunk.compute(result)
    #print(bipartite_pairings)
    pal_clr_list_lab =[]
    #print('current_image_colors', current_image_colors)
    for im, ds in bipartite_pairings:
        col= pal_clr_list_lab_no_weight[ds]
        pal_clr_list_lab.append((col[0], col[1], col[2], current_image_clusters[im][3]))
    print('pal_clr_list_lab', pal_clr_list_lab)
    #render_palette_horizontal_weights(pal_clr_list_lab,   os.path.join(d, "actual_DS_pal"), "LAB", 500)
    #print('current_image', current_image)
    return current_image,  current_image_clusters, pal_clr_list_lab

def matching_algorithm_new_weighted_wrapper(pal_clr_list_lab,used_css_rgb_weighted, L_only, webpath, d, old_bg, css_bg):
    final_matching =[]
    matching_with_centroid ={}
    new_bg=[]
    if len(used_css_rgb_weighted)>0:
        final_matching, matching_with_centroid=matching_algorithm_new_weighted(pal_clr_list_lab,used_css_rgb_weighted, L_only, webpath, d)
        print('final_matching', final_matching)
        print('old_bg', old_bg)
        new_bg = final_matching[old_bg]
        # new_bg_rgb = convert_color(new_bg_lab, AdobeRGBColor) 
        # new_bg = new_bg_rgb.get_upscaled_value_tuple()
        #final_matching, new_bg= match_bg_lightness(old_bg, css_bg, final_matching, d)
    return final_matching, new_bg, matching_with_centroid

from Chameleon.image_recolor import LAB2RGB1, RGB2LAB
def recolor_all_recolorable_wrapper(recol_path, pal_clr_list_lab, old_bg, new_bg):
    print('pal_clr_list_lab, old_bg, new_bg', pal_clr_list_lab, old_bg, new_bg)
    #need to convert current_image_clusters to RGB w/o weights
    lablist = [LabColor(l1,l2,l3) for l1,l2,l3,w in pal_clr_list_lab]
    target_palette = [convert_color(x, AdobeRGBColor) for x in lablist]
    target_palette = [x.get_upscaled_value_tuple() for x in target_palette]
    #print('target_palette', target_palette)
    recolor_all_recolorable(recol_path, target_palette, pal_clr_list_lab, old_bg, new_bg)
    #print('recolor_all_recolorable_wrapper done')

def recolor_css_new_wrapper(final_matching, css_names, webpath):
    #for testing
    # final_matching ={}
    # final_matching_file = open(os.path.join(webpath, 'final_matching'), "rb")
    # try:
    #     final_matching = pickle.load(final_matching_file)
    # except:
    #     print('Error reading file or file is empty!')
    # print('final_matching', len(final_matching), final_matching)

    # css_names, html_names = find_css(webpath, 1) #1 is to save a copy
    #print(css_names)
    if len(final_matching)>0:
        replaced=recolor_css(final_matching, css_names)
    #print('replaced', replaced)
    #recolor_css_new(final_matching, css_names, webpath)

def match_bg_lightness(old_bg, css_bg, final_matching, d):
    old_bg_lab = RGB2LAB(old_bg)#convert_color(old_bg_rgb, LabColor)
    old_bg_matching =[]
    new_bg_matching =[]
    new_bg =()
    for c,s in css_bg: #c is the original css value in RGB
        #new_bg_rgb = AdobeRGBColor(final_matching[old_bg][0], final_matching[old_bg][1], final_matching[old_bg][2])
        new_bg_lab = RGB2LAB(final_matching[c])#convert_color(new_bg_rgb, LabColor) 
        new_bg_lab.lab_l= old_bg_lab.lab_l
        new_bg_rgb = LAB2RGB1(new_bg_lab)#convert_color(new_bg_lab, AdobeRGBColor)
        new_bg = new_bg_rgb#.get_upscaled_value_tuple()#(new_bg_rgb.rgb_r, new_bg_rgb.rgb_g, new_bg_rgb.rgb_b)
        print('old_bg', c, 'final_matching[old_bg]', final_matching[c], 'new_bg', new_bg)
        old_bg_matching.append((c, final_matching[c]))
        final_matching[c] = new_bg
        new_bg_matching.append((c, new_bg))
    render_matching(sorted(old_bg_matching, key=lambda x:x[0]), os.path.join(d,  "old_bg_matching"), 'rgb')
    render_matching(sorted(new_bg_matching, key=lambda x:x[0]), os.path.join(d,  "new_bg_matching"), 'rgb')

    # new_bg = [v for k,v in final_matching.items() if k == old_bg]
    # print('new_bg', new_bg)
    # new_rgb_bg =()
    # if len(new_bg)>0:
    #     new_rgb_bg = new_bg[0]
    # else:
    #     print('ERROR: CANNOT FIND BACKGROUND')
    return final_matching, new_bg


def render_results(webpath, final_matching, pal_clr_list_lab, d, html_name, matching_with_centroid):
    scurl ='file://'+os.path.join(webpath,html_name)#'index.html')
    take_screenshot_selenium(scurl, os.path.join(d , "out.png"))   
    #final_matching - final old:new
    #print('final_matching.values()', final_matching.values())
    if len(matching_with_centroid)==0 and len(final_matching)>0:
        #this ensures that they are in the same order but it doesn't look good
        final_matching_list = [(k,v) for k,v, in final_matching.items()]
        final_matching_list_sorted = [(k,v) for k,v in sorted(final_matching_list, key=lambda x:x[0])]
        render_palette([v for k,v in final_matching_list_sorted],   os.path.join(d, "out_pal"), "rgb")
        render_palette([k for k,v in final_matching_list_sorted],   os.path.join(d, "original_pal"), "rgb")
        # final_matching_list = [(k,v) for k,v, in final_matching.items()]
        # render_matching(sorted(final_matching_list, key=lambda x:x[0]), os.path.join(d,  "matching"), 'rgb')
        # render_matching(sorted(final_matching_list, key=lambda x:x[1]), os.path.join(d,  "matching1"), 'rgb')
    if len(matching_with_centroid)>0:
        #this ensures that they are in the same order but it doesn't look good
        final_matching_list = [(k,v[1]) for k,v, in matching_with_centroid.items()]
        final_matching_list_sorted = [(k,v) for k,v in sorted(final_matching_list, key=lambda x:(x[1], x[0].lab_l))]
        render_palette([k for k,v in final_matching_list_sorted],   os.path.join(d, "out_pal"), "LAB")
        final_matching_list = [(v[0],v[1]) for k,v, in matching_with_centroid.items()]
        final_matching_list_sorted = [(k,v) for k,v in sorted(final_matching_list, key=lambda x:(x[1], x[0][0]))]
        render_palette([k for k,v in final_matching_list_sorted],   os.path.join(d, "original_pal"), "LAB")

    #print('pal_clr_list_lab', pal_clr_list_lab)
    #render_palette_horizontal_weights([((c1, c2, c3), w) for c1, c2, c3, w in final_matching.values()],   os.path.join(d, "out_pal_im"), "LAB", 500)
    #render_palette_horizontal_weights([((c1, c2, c3), w) for c1, c2, c3, w in current_image_clusters],   os.path.join(d, "out_pal_im"), "LAB", 500)
    if len(pal_clr_list_lab)>0:
        render_palette_vertical_weights([((c1, c2, c3), w) for c1, c2, c3, w in pal_clr_list_lab],   os.path.join(d, "out_pal_im"), "LAB", 500)
        render_palette_vertical([(c1, c2, c3) for c1, c2, c3, w in sorted(pal_clr_list_lab, key =lambda x: (x[1], x[2]))],   os.path.join(d, "out_pal_im"), "LAB")

#end do_weighted_matching_full_path pieces
##############################################################################


def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def test():
    final_matching ={}
    final_matching_file = open(os.path.join("/home/linka/python/autoimage_flask/uploads/catalina/code/code", 'final_matching'), "rb")
    try:
        final_matching = pickle.load(final_matching_file)
    except:
        print('Error reading file or file is empty!')
    print('final_matching', len(final_matching), final_matching)
    #final matching: orig hex(css) to new hex (palette)
    #(231, 157, 59): (240, 153, 57)}


if __name__ == "__main__":
    print( 'Started')
    print('Clustered test image')
    clusterfilename= "backend/clusterlistbytesweighted.txt"
    training_clusters = deserialize(clusterfilename)
    for c in training_clusters:
        print( c)
        break
    # #get_weighted_palettes()
    # #read_weighted_palettes()

    # #test_weighted_matching()
    # #test_recolor_css("/home/linka/python/autoimage_flask/uploads/code/code/")
    # #find_nearest_cluster_weighted('/home/linka/python/autoimage_flask/uploads/code/auroreboreale_by_tohad-d8k9co6.jpg', "/home/linka/python/autoimage_flask/uploads/code/")
    # #test_create_additional_colors("/home/linka/python/autoimage_flask/uploads/test1")
    # #test()
    # #evaluate_weight_lambda()
    # #test_wo_lambda()
    # #test_replace_in_css("/home/linka/python/autoimage_flask/uploads/test/")
    # #compare_weighted_unweighted_matching()
    # tmpfolder ='/home/linka/python/autoimage_flask/uploads/tmp2ap9a58x/'
    # url ="http://python-3-patterns-idioms-test.readthedocs.io/en/latest/Comprehensions.html"#http://www.kccreepfest.com/"#"http://cge.fsu.edu/newStudents/applyingtofsu.html"#'www.cs.fsu.edu'
    # user_image= 'g_family_33_by_livingwild-d9hy581.jpg'#decay_by_angelarizza-d82r4cp.jpg'
    # d =tmpfolder
    # # #download_and_extract_palette_wrapper(url, dfind_nearest_cluster_weighted(starting_filename, dest_path), d)
    # #do_weighted_matching_full_path(url,'', user_image, d) #(url, web_archive, image_name, tmpfolder)

    # """
    # #for bronte
    # tmpfolder ='/var/www/autoimage/uploads/bg/' #'/home/linka/python/autoimage_flask/uploads/tmp539m1f18/'
    # url ="https://www.rei.com/"#"http://cge.fsu.edu/newStudents/applyingtofsu.html"#'www.cs.fsu.edu'
    # user_image= 'i3.png'#decay_by_angelarizza-d82r4cp.jpg'
    # d ="//var/www/autoimage/uploads/bg/"
    # # #download_and_extract_palette_wrapper(url, dfind_nearest_cluster_weighted(starting_filename, dest_path), d)
    # do_weighted_matching_full_path(url,'', user_image, d) #(url, web_archive, image_name, tmpfolder)
    # """
    # test_bipartite_match_css_to_palette()
    # print('Done')