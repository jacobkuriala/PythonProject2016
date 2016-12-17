'''
Created on March 5, 2016

@author: Polina

Contains methods for weighted palette extraction from website 
- Css extraction is based on screenshot with no images, and css files
- Image palette extractions clusters colors from all images 
that are classified as recolorable
'''
import subprocess
def download_website(url, path):
    #args = ['wget', '--no-directories', '--no-clobber', '--no-parent', '--convert-links', '--page-requisites', '-E',  '-e', 'robots=off', '-k', '-P', path , url]
    args = ['wget', '--no-directories', '--no-clobber', '--no-parent', '--convert-links', '--page-requisites', '-E',  '-e', 'robots=off', '-k', '-H', '-P',  path , url]
    subprocess.call(args, timeout=3000)


def allowed_file(fileusername, extensions):
    # if '.' in fileusername: 
    #     fileName,fileExtension = os.path.splitext(fileusername)
    #     print( fileName,fileExtension)
    #     return fileExtension.lower() in extensions
    #     #return (fileusername.rsplit('.', 1)[1]).lower() in extensions
    #the above doesn't work for cases like pic.jpg.1 which is a valid jpg
    result = 0
    for e in extensions:
        if fileusername.endswith(e) or e+'.' in fileusername:
            result =1
            break
    return result


import time
def download_and_extract_palette(url, path, dest_path):
    """
    1) download website
    2) identify all images
    3) remove all videos
    4) classify images into recolorable/nonrecolorable
    5) move nonrecolorable, take screenshot
    return list of recolorable
    """
    #1
    #url ='http://www.theonion.com/article/obama-resigns-presidency-after-michelle-lands-drea-52338'# 'http://www.msn.com/en-us/video/animals/raw-rare-white-falcon-caught-on-camera-courtesy-exploreorg/vi-BBpmzxI?ocid=UP97DHP'
    # url ='http://www.cs.fsu.edu/'#'http://www.cs.fsu.edu/department/faculty/sudhir/'# 
    # path = 'code'
    threshold = 20
    download_website(url, path)    
    #2    
    used_css_rgb_weighted=[]
    recolorable_image_palette=[]
    now1 = time.time()
    unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba=get_css_palette(path, '','')#(webfolder, imagefolder, image_name)
    #print('unique_css_HEX',unique_css_HEX)
    now2 = time.time()
    print('get_css_palette took', now2 - now1)
    classify_images(url, path, dest_path, html_name)
    now3 = time.time()
    print('classify_images took', now3 - now2)
    if len(unique_css_HEX)>0 or len(unique_css_rgba)>0:
        used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, bg_rgb, css_bg=get_full_palette_css_img(path, threshold, unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba)
    else:
        print("NO COLORS ON THE WEBPAGE???")
    now4 = time.time()
    print('get_full_palette_css_img took', now4 - now3)
    #copy nonrecolorable back
    nrpath =join(path,'non-recolorable')
    #print(nrpath, 'nrpath')
    if exists(nrpath):
        nrfiles = [f for f in listdir(nrpath) if isfile(join(nrpath, f))]
        for f in  nrfiles:
            shutil.copy(join(nrpath, f), join(path, f))  
            #print('copy', join(nrpath, f), join(path, f))
    return used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, bg_rgb, css_bg


from pyvirtualdisplay import Display
from selenium import webdriver
def take_screenshot_selenium(url, dest):
    display = Display(visible=0)#, size=(800, 600))
    display.start()
    browser = None
    browser = webdriver.Firefox()

    browser.get(url)
    browser.maximize_window()
    browser.save_screenshot(dest)
    browser.quit()
    display.stop()
    print('Screenshot done', url, dest)

import shutil
from os import listdir, getcwd, mkdir, remove
from os.path import isfile, join, exists
from Chameleon.image_recolor_classifier import get_model, predict_recolorability_use_model
def classify_images(url, path, dest_path, html_name):
    """
    1) download website
    2) identify all images
    3) remove all videos
    4) classify images into recolorable/nonrecolorable
    5) move nonrecolorable, take screenshot
    move  recolorable
    take screenshot
    return list of recolorable
    """
    #time1 = time.time()
    IMAGE_EXTENSIONS = set(['.png', '.jpg', '.jpeg', '.gif'])
    images =[]
    images1 =[] #recolorable
    images0 =[] #nonrecolorable
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    #4 classify images into recolorable/nonrecolorable
    model= get_model('img_recolor_model')
    for f in  onlyfiles:
        #print(f)
        if allowed_file(f, IMAGE_EXTENSIONS):
            images.append(f)
            res = predict_recolorability_use_model(path,f, model)
            #print(res)
            if res ==1:
                images1.append(f)
            else:
                images0.append(f)   
    #time2 = time.time()
    #print('classification took', time2-time1)
    #browser.get('file:///home/linka/python/autoimage_flask/uploads/tmpzk6taiv5/code/index.html')
    #take_screenshot_not_selenium(join(path,'index.html'), join(path,'original.png'))
    scurl ='file://'+join( getcwd(),path,html_name) #'index.html')
    #print(scurl)
    take_screenshot_selenium(scurl, join(dest_path , "original.png"))
    #time3 = time.time()
    #print('screenshot took', time3-time2)
    # print('recolorable', images1)
    # print('non-recolorable', images0)
    #3 if needed
    #5) move nonrecolorable, take screenshot
    dest = join(getcwd(), path,'non-recolorable')
    if not exists(dest):
        mkdir(dest)
    for p in images0:
        #print(join(getcwd(),path, p), dest)
        try:
            shutil.move( join(getcwd(), path, p), dest)  
            remove(join( path, p))
        except:
            pass
    #take_screenshot_not_selenium(join(path,'index.html'), join(path,'recol_only.png'))
    #we're not using this so don't waste time
    #take_screenshot_selenium(scurl, join(path,'recol_only.png'))
    
    dest = join(getcwd(), path,'recolorable')
    if not  exists(dest):
        mkdir(dest)
    for p in images1:
        #print(join(getcwd(),path, p), dest)
        try:
            shutil.move( join(getcwd(), path, p), dest)  
            remove(join( path, p))
        except:
            pass
    #take_screenshot_not_selenium(join(path,'index.html'), join(path,'no_img.png'))
    #time4 = time.time()
    #print('file copy took', time4-time3)
    take_screenshot_selenium(scurl, join(path,'no_img.png'))
    #time5 = time.time()
    #print('screenshot took', time5-time4)

def copy_nonrecolorable_back(path): 
    """
    path is the folder  where index.html is located
    """
    src = join(getcwd(), path,'non-recolorable')
    dest = join(getcwd(), path)
    #print('dest, src',dest, src)
    if exists(src):
        onlyfiles = [f for f in listdir(src) if isfile(join(src, f))]
        for f in  onlyfiles:
            try:
                #print('moving', join(src, f), dest)
                shutil.move( join(src, f), dest)  
            except:
                pass

from PIL import Image
from Chameleon.website_color_process import get_css_palette, hex_to_rgb, render_palette_vertical, render_matching, render_palette_vertical_weights
from Chameleon.image_palette_process import remove_duplicate_palette_colors
def get_css_palette_from_css_and_screenshot(path, threshold, unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba, recolorbw = 1):
    """
    get all css colors
    get all colors from screenshot w/o images
    pick css colors that occur in screenshot
    recolorbw - are we going to replace black and white or not. 
    default to yes cannot justify not replacing
    """
    print('unique_css_HEX', len(unique_css_HEX)) #HEX
    print('unique_css_rgba', len(unique_css_rgba))#, unique_css_rgba) #HEX
    #to rgb
    css_rgb ={}# [ hex_to_rgb_(k) for k in unique_css_HEX]
    for k in unique_css_HEX:
        hex_ = hex_to_rgb_(k)
        if hex_!=None:
            if not recolorbw:
                if hex_!=( 255,255,255)and hex_!=( 255,255,255):
                    css_rgb[hex_]= k
                    #css_rgb[k]= hex_to_rgb_(k)
                else:
                    print('recolorbw',recolorbw, hex_to_rgb_(k), 'excluded')
            else:
                css_rgb[hex_]= k
    for k in unique_css_rgba:
        rgb = (k[0], k[1], k[2])
        if rgb!=None:
            if not recolorbw:
                if rgb!=( 255,255,255)and rgb!=( 255,255,255):
                    css_rgb[rgb]= k
                    #css_rgb[k]= hex_to_rgb_(k)
                else:
                    print('recolorbw',recolorbw, rgb, 'excluded')
            else:
                css_rgb[rgb]= k
    #print('css_rgb', len(css_rgb), css_rgb)
    screenpalette =[]
    try:
        im = Image.open(join(path,'no_img.png'))
        im = im.convert('P', palette=Image.ADAPTIVE).convert("RGB")
        screenpalette = im.getcolors()#getpalette() #RGB
    except:
        print('Cannot find screenshot no_img.png')
    # width, height =im.size
    # #different threshold for every image based on size
    # color_threshold = width*height*0.001#0.005
    #discard small values, and then clean out the duplicates
    screen_rgb =[]
    screen_rgb_with_weights ={}

    #from screenshot, pick the color that appears most (as background!)
    bg_rgb =()
    bg_weight = 0
    for c, rgb in screenpalette:
        #if c> color_threshold: #nope, we want all css
        screen_rgb_with_weights[rgb] =c
        screen_rgb.append(rgb)
        if c>bg_weight:
            bg_weight =c
            bg_rgb = rgb # this is really just the color that appears most in the screenshot

    # print('before dupl remove', len(screen_rgb))
    # screen_rgb=remove_duplicate_palette_colors(screen_rgb, threshold)
    # print('after dupl remove', len(screen_rgb))
    print('screen_rgb',len(screen_rgb))
    #render both
    #render_palette_vertical(screen_rgb,  join(path,"screencss_no_dupl"), "rgb")
    
    css_rgbonly =[k for k in css_rgb.keys()] #[r for r,k in css_rgb]
    render_palette_vertical(sorted(css_rgbonly), join(path,"css"), "rgb")
    #exclude white/black?
    #find matches
    used_css_rgb_values, used_css_rgb_weighted, matching=match_palette_colors(css_rgbonly, screen_rgb, screen_rgb_with_weights,  threshold)
    ################ backgroud ###############
    # bg_elements = [ [k,v] for k,v in result_elems  if 'background' in k]
    # print('bg_elements',bg_elements)
    # bg_elements_rgba = [ [k,v] for k,v in result_elems_rgba  if 'background' in k]
    # print('bg_elements_rgba',bg_elements_rgba)
    # css_rgb_bg = [ r for r, h in css_rgb.items() if h in [ v for k,v in bg_elements]]
    # print('css_rgb_bg',css_rgb_bg)
    # render_palette_vertical(sorted(css_rgb_bg), join(path,"css_bg_colors"), "rgb")

    print('screen background', bg_rgb, bg_weight)
    old_bg_rgb = bg_rgb
    #find closest color to background
    #[((242, 222, 222), (251, 215, 232)), css:screen
    #css_bg =[] # decided not to do bg lightness matching    
    css_bg= [(c, s) for c,s in matching if s == bg_rgb]
    #render_matching(css_bg_, join(path,"css_to_screen_bg"), "rgb")
    # #now olny pick actual bg elements - exactly same so maybe all of the above is completely unnecessary
    # css_bg = [(c, s) for c,s in css_bg_ if c in css_rgb_bg]
    # render_matching(css_bg_, join(path,"css_to_screen_bg_only"), "rgb")
    print('css_bg', css_bg)
    if len(css_bg)>0:
        #pick one with biggest weight
        maxw =0
        for c,s in css_bg:            
            for r, w in used_css_rgb_weighted:
                if r ==c and w> maxw:
                    #print(r,w, c, maxw)
                    maxw =w
                    bg_rgb =c

    render_matching([(bg_rgb, old_bg_rgb)], join(path,"old_to_new_bg"), "rgb")
    ################ end backgroud ###############
    #check if it's in background element - nah
    # print('result_elems', [r for r in result_elems if 'background' in r[0]])
    # print('result_elems_rgba', [r for r in result_elems_rgba if 'background' in r[0]] )

    # total_used = sum(w for w in used_css_rgb_weighted.values())
    # print('total_used', total_used)
    print('screen_rgb_with_weights', len(screen_rgb_with_weights))
    #print('used_css_rgb_values', used_css_rgb_values)
    #used_css_rgb_values, matching=match_palette_colors(css_rgbonly, screen_rgb, threshold)
    render_matching(matching, join(path,"css_to_screen_css"), "rgb")
    render_palette_vertical(sorted(used_css_rgb_values), join(path,"used_css"), "rgb")
    #[(a,c) for a,b,c in sorted(extended_palette, key=lambda x: x[1])]
    render_palette_vertical_weights([(a,c) for a,c in sorted(used_css_rgb_weighted, key=lambda x: x[0])], join(path,"used_css_weighted"), "rgb", 500)
    # print('used_css', len(used_css_rgb_values))
    # print('used_css_rgb_weighted', used_css_rgb_weighted[0])
    used_css_hex_values =set([k for r,k in css_rgb.items() if r in used_css_rgb_values])
    print('here test4', used_css_hex_values)
    render_palette_vertical(used_css_hex_values, join(path,"used_css_hex"), "hex")
    print('used_css_hex_values', len(used_css_hex_values))#, used_css_hex_values)
    return used_css_rgb_values , used_css_hex_values, used_css_rgb_weighted, bg_rgb, css_bg#, css_names, html_name


def get_full_palette_from_screenshot(path, threshold):
    """
    get all colors from screenshot with images
    """
    im = Image.open(join(path,'recol_only.png'))
    im = im.convert('P', palette=Image.ADAPTIVE).convert("RGB")
    screenpalette = im.getcolors()#getpalette() #RGB
    #discard small values, and then clean out the duplicates

    screen_rgb =[]
    screen_rgb_with_weights =[]
    total_count = 0
    for c, rgb in screenpalette:
        if c> threshold:
            screen_rgb_with_weights.append((c,rgb))
            screen_rgb.append(rgb)
        total_count = total_count+c
    print('before dupl remove', len(screen_rgb))
    remove_duplicate_palette_colors(screen_rgb, threshold)
    print('after dupl remove', len(screen_rgb))
    #render both
    render_palette_vertical(screen_rgb,  join(path,"screen_full"), "rgb")
    return screen_rgb

def get_full_palette_css_img(path, threshold, unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba):
    # path = 'code'
    # threshold = 20
    used_css_rgb_values, used_css_hex_values, used_css_rgb_weighted, bg_rgb, css_bg=get_css_palette_from_css_and_screenshot(path, threshold, unique_css_HEX, result_elems, css_names, html_name, unique_css_rgba, result_elems_rgba)
    #screen_rgb = get_full_palette_from_screenshot(path, threshold)
    #we're not really using the image palette
    #recolorable_image_palette=get_recolorable_palette(path, threshold)
    recolorable_image_palette=[]
    #which are css and which are img 0- remove css colors from image colors
    # only_screen_rgb = remove_close_palette_colors(screen_rgb, used_css_rgb_values, threshold)
    # print(len(only_screen_rgb), 'only_screen_rgb')
    # render_palette_vertical(only_screen_rgb, join(path,"only_screen_rgb"), "rgb")
    # print(len(used_css_rgb_values), 'used_css_rgb_values')   
    # #combine 
    # full_palette = only_screen_rgb
    # full_palette=full_palette+ used_css_rgb_values
    # #weights?
    # print(len(full_palette), 'full_palette')
    #render_palette_vertical(full_palette, join(path,"full_palette"), "rgb")
    return used_css_rgb_weighted, recolorable_image_palette, css_names, html_name, bg_rgb, css_bg

def remove_close_palette_colors(remove_from, to_remove, threshold):
    """
    Calculate eucledian distance between colors
    Where distance<threshold, assume it's the same color
    """
    dist_matrix = create_distance_matrix(to_remove, remove_from)
    #print(dist_matrix)
    #print(len(rgb_values))
    clean =remove_from
    duplicates=[]
    for j in range(0,len(to_remove)):
        for k in range(0,len(remove_from)):
            if k>j and dist_matrix[j][k]>threshold:
                if remove_from[k] not in duplicates:
                    duplicates.append(remove_from[k])
    #print(duplicates)
    render_palette_vertical(duplicates, join("code","duplicates"), "rgb")
    for d in duplicates:
        if d in clean:
            clean.remove(d)
    return clean

def hex_to_rgb_(k):
    if '#' in k:
        rgb =hex_to_rgb(k[1:])
    else:
        rgb =hex_to_rgb(k)
    return rgb

from Chameleon.nearest_palette import create_distance_matrix
def match_palette_colors(css_values, screen_values, screen_values_weighted, threshold):
    """
    we assume that len(css_values)>=len(screen_values) -? not necessarily
    Calculate eucledian distance between colors
    Where distance<threshold, assume it's the same color
    """
    if len(css_values)<len(screen_values):
        print('Violated assumption: len(css_values)<len(screen_values)', len(css_values),len(screen_values))
    # total_screen = sum(w for w in screen_values_weighted.values())
    print('screen_values', [v for v in screen_values if len(v)>3])
    print('css_values', [v for v in css_values if len(v)>3])
    dist_matrix = create_distance_matrix(screen_values, css_values)
    #print('dist_matrix', dist_matrix)
    used_css_rgb_values =[]
    used_css_rgb_weighted ={}
    matching =[]
    #find closest css to screen
    for j in range(0,len(screen_values)):
        for k in range(0,len(css_values)):
            if dist_matrix[j][k]<threshold:
                if css_values[k] not in used_css_rgb_values:
                    used_css_rgb_values.append(css_values[k])
                    used_css_rgb_weighted[css_values[k]]=screen_values_weighted[screen_values[j]]#int(screen_values_weighted[screen_values[j]]*100/total_screen)
                    matching.append((css_values[k], screen_values[j]))
                    #print('new', used_css_rgb_weighted[css_values[k]],screen_values_weighted[screen_values[j]])#, int(screen_values_weighted[screen_values[j]]*100/total_screen))
                else:
                    #print('update', used_css_rgb_weighted[css_values[k]], screen_values_weighted[screen_values[j]])#, int(screen_values_weighted[screen_values[j]]*100/total_screen))
                    used_css_rgb_weighted[css_values[k]]=used_css_rgb_weighted[css_values[k]]+screen_values_weighted[screen_values[j]] #int(screen_values_weighted[screen_values[j]]*100/total_screen)
                    #print('updated', used_css_rgb_weighted[css_values[k]])
    #if 3 css colors match to same screen color, they should not get same % - should get 1/3 of it
    for sc in  screen_values:
        #count matches
        cnt = [r for h, r in matching].count(sc)
        #print('sc, cnt', sc, cnt)
        for hx, sc in [(h,s) for h,s in matching if s==sc]:
            #print( hx, sc, 'before',  used_css_rgb_weighted[hx] ,'/',cnt)
            used_css_rgb_weighted[hx] = used_css_rgb_weighted[hx]/cnt
            #print('after', used_css_rgb_weighted[hx])
    #do not allow % <1
    total_screen = sum(w for w in used_css_rgb_weighted.values())
    print('total_screen', total_screen)
    #convert weights to percentages
    used_css_rgb_weighted = {k:int(w*100/total_screen) for k,w in used_css_rgb_weighted.items()}
    
    total_screen_perc = sum(w for k,w in used_css_rgb_weighted.items())
    print('total_screen_perc', total_screen_perc)
    for  k,w in used_css_rgb_weighted.items():
        if w==0:
            used_css_rgb_weighted[k] =1
    total_screen_perc = sum(w for k,w in used_css_rgb_weighted.items())
    print('total_screen_perc', total_screen_perc)
    return used_css_rgb_values, [(k,w) for k,w in used_css_rgb_weighted.items()], matching

def get_recolorable_palette(path, threshold):
    """
    get all colors from images in recolorable folder?
    """
    colors =[]
    colors_with_weights =[]
    total_count = 0
    for f in  listdir(join(path, 'recolorable')):
        #print(f)
        im = Image.open(join(path,'recolorable', f))
        im = im.convert('P', palette=Image.ADAPTIVE).convert("RGB")
        screenpalette = im.getcolors()#getpalette() #RGB
        width, height =im.size
        #different threshold for every image based on size
        color_threshold = width*height*0.001#0.005
        #print('color_threshold', color_threshold, width,height, width*height)
        for c, rgb in screenpalette:
            if c> color_threshold:
                if rgb not in colors:
                    colors.append(rgb)
                    colors_with_weights.append((c,rgb))
            total_count = total_count+c

    print('before dupl remove', len(colors))
    remove_duplicate_palette_colors(colors, threshold)
    print('after dupl remove', len(colors))
    #render both
    render_palette_vertical(colors,  join(path,"recolorable"), "rgb")
    return colors

from cluster import cluster_multiple_images
from numpy import bincount
def get_recolorable_palette_clustering(path, threshold):
    """
    get all colors from images in recolorable folder?
    """
    images = []
    for f in  listdir(join(path, 'recolorable')):
        print(f)
        im = Image.open(join(path,'recolorable', f))
        im = im.convert('P', palette=Image.ADAPTIVE).convert("RGB")
        images.append(im)
    labpalette, labels = cluster_multiple_images(images, k_clusters=5)
    print('labpalette',labpalette)
    print('labels',labels.shape)

    labpalette_withweights =[] # weight is % rounded to int. like 50 (%)
    weights = bincount(labels)
    print('weights', weights)
    total = sum(weights)
    print('total', total)
    i=0
    for lab in labpalette:
        labpalette_withweights.append((lab, int(weights[i]*100/total)))
        print(lab, int(weights[i]*100/total))
        i =i+1

    print(labpalette_withweights)
    #render both
    render_palette_vertical(labpalette,  join(path,"labcluster"), "LAB")
    render_palette_vertical_weights(labpalette_withweights,   join(path,"labcluster_weighted"), "LAB")
    return labpalette_withweights



if __name__ == "__main__":
    print( 'Started')
    # #get_full_palette_from_screenshot()
    # #get_full_palette_css_img()
    # #download_and_extract_palette()
    # #get_recolorable_palette_clustering('code', 20)
    # #get_css_palette_from_css_and_screenshot('code', 20)
    # #copy_nonrecolorable_back('code')
    # #url = "http://cge.fsu.edu/newStudents/applyingtofsu.html"
    # url ="www.rei.com"
    # dest = "/uploads/bg/"
    # #dest = "/home/linka/python/autoimage_flask/uploads/test4/"
    # take_screenshot_selenium(url, dest)
    # #download_website(url, dest)
    # download_and_extract_palette(url, dest, dest)
    classify_images("www.rei.com", "/home/linka/python/autoimage_flask/uploads/tmpfrye5qn0/code/", "/home/linka/python/autoimage_flask/uploads/tmpfrye5qn0/code/", "index.html")
    print( 'Done')