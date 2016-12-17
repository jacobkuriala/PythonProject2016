import numpy as np
import featurex as fx
import featureexhelpers as fxh
import re

img = np.array([ [ [1, 2, 3], [5, 7, 8], [1,2,3] ], [ [5, 7, 8], [5, 7, 8] ] ])
k_means_palette = np.array([[4, 5, 6], [4, 5, 6]])
lab_img = [list(row) for row in img[0]]
b = k_means_palette[:,1]
u = [list(x) for x in set(tuple(x) for x in lab_img)]
print(u)
beached = "BeachedBright.png:[[0.026983979284493403, 0.010259214301016811, 0.0053510651682631928],\
            [0.026182867057888123, 0.010933184210628687, 0.007250856451522499], \
            [0.023590198683117045, 0.011564842336081576, 0.010186064169044454], \
            [0.024878741035997947, 0.0096353887313127967, 0.0053011847390608104], \
            [0.018338263718691853, 0.011921749875351951, 0.012022082523392869], \
            [0.01576844102185844, 0.011110646284339096, 0.011311000634383879]]\n"

beached3 = "BeachedBright3.png:[[0.036983979284493403, 0.010259214301016811, 0.0053510651682631928],\
            [0.026182867057888123, 0.010933184210628687, 0.007250856451522499], \
            [0.023590198683117045, 0.011564842336081576, 0.010186064169044454], \
            [0.024878741035997947, 0.0096353887313127967, 0.0053011847390608104], \
            [0.018338263718691853, 0.011921749875351951, 0.012022082523392869], \
            [0.01576844102185844, 0.011110646284339096, 0.011311000634383879]]"

beached4= "BeachedBright.png:[[61.035925820389039, 45.196762508117885, 60.387491133389346],\
                   [81.32901768458018, 11.479369527211292, 50.758496905780625],\
                   [90.655978231128259, 0.97750881396668765, 24.290748015727569],\
                   [87.539194744031121, -2.3554643260915165, 11.303323293271017],\
                   [94.340716482252958, -3.2386469237509763, 9.1892245951778584],\
                   [52.453627366354382, 50.660176284820501, 56.805366475710748]]"

generated_palette = np.array([[78.52803202440613, -18.04511071780229, 1.4561559380717348],
 [67.53368952483828, 8.00519582819642, 9.943631988166857],
 [53.880699807832485, 2.0535429000175243, -5.384156729971079],
 [79.87240965733825, -11.062011146226125, 3.035484006758682],
 [86.41624532050963, 4.124204646322072, 9.789257035327893]])
print(generated_palette[:, 0])
roguefilelist = ["BeachedBright.png"]
img_dict = {}
pixel_pattern = '\[-*[0-9]+\.[0-9]+, -*[0-9]+\.[0-9]+, -*[0-9]+\.[0-9]+\]'
float_pattern = '-*[0-9]+\.[0-9]+'

for p in range(2):
    if(p == 0):
        name, pixel_list = beached.split(':')
    else:
        name, pixel_list = beached3.split(':')
        for pixel in re.findall(r'' + pixel_pattern, pixel_list):
            img_dict[name] = list(map(float, re.findall(r'' + float_pattern, pixel)))

print(img_dict)

#if name not in roguefilelist:
#    palette_dict[name] = re.findal(r'' + pixel_pattern, pixel_list)

"""
weight = 1 / 2
total1 = 0
for p in lab_img:
    total = 0
    print("Pixel" + str(p))
    for c in k_means_palette:
        print("Swatch" + str(c))
        upc = fxh.calculate_upc(p, c, k_means_palette)**2
        error = fxh.calculate_euclid_dist(p, c)**2
        print("UPC:" + str(upc))
        print("Error:" + str(error))
        total += (upc * error)
        print("Total for color " + str(c) + " : "+ str(total))
        print()
    total1 += total * weight
    print("Total times weight: " + str(total1))
    print()

print(fx.extract_soft_recoloring_error(lab_img, k_means_palette))
"""
