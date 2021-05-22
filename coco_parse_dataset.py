import json

import skimage
from skimage import io
import matplotlib.pyplot as plt
from skimage.draw import rectangle
from skimage.draw import rectangle_perimeter, set_color
from skimage import filters
from skimage.measure import label
import numpy as np
from skimage.color import rgb2gray

#path_m = 'task_mach2020-27-12/'
path_m = 'task_mach2021-02-20/'
path_m = 'task_mach2021-02-25/'

path_json = path_m + 'annotations/instances_default.json'
# annotation = json.loads()
with open(path_json, "r") as read_file:
    data = json.load(read_file)



path_img_target = 'anot/True/'
path_img_targer_false =  'anot/False/'

for num in range(0,len(data['images'])):

    found = True

    name_i = data['images'][num]['file_name']
    img_id = data['images'][num]['id']
    path_images = path_m + '/images/' + name_i

    print(path_images)
    img = io.imread(path_images)
    y_max,x_max, c = img.shape

    # print(img)
    k=0

    for i in range(0, len(data['annotations'])):
        if data['annotations'][i]['image_id'] == img_id:

            found = False

            print(data['annotations'][i]['bbox'])
            print(data['annotations'][i]['area'])
            x0, y0, width, height = data['annotations'][i]['bbox']
            y0 = round(y0)
            x0 = round(x0)
            width = round(width)
            height = round(height)



            start = (y0, x0)
            end = (y0 + height, x0 + width)

            rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)
            #set_color(img, (rr, cc), (255, 0, 0))

            cropped = img[y0:y0 + height, x0:x0 + width]
            io.imsave(path_img_target + name_i + '_' + str(i) + '.jpg', cropped)

            y_in = +50
            x_in = +50


            if y0>50 and x0>50 and y0<y_max-50 and x0<x_max-50:

                cropped_bad_1 = img[y0 + y_in:y0 + height + y_in, x0 + x_in:x0 + width + x_in]
                io.imsave(path_img_targer_false + name_i + '_false_' + str(k) + '.jpg', cropped_bad_1)
                k=k+1
                cropped_bad_2 = img[y0 - y_in:y0 + height - y_in, x0 + x_in:x0 + width + x_in]
                io.imsave(path_img_targer_false+ name_i + '_false_' + str(k) + '.jpg', cropped_bad_2)
                k=k+1
                cropped_bad_3 = img[y0 + y_in:y0 + height + y_in, x0 - x_in:x0 + width - x_in]
                io.imsave(path_img_targer_false + name_i + '_false_' + str(k) + '.jpg', cropped_bad_3)
                k=k+1
                cropped_bad_4 = img[y0 - y_in:y0 + height - y_in, x0 - x_in:x0 + width - x_in]
                io.imsave(path_img_targer_false + name_i + '_false_' + str(k) + '.jpg', cropped_bad_4)
                k=k+1

    print(found)

    if found:
        if (path_m == 'task_mach2021-02-25/' or path_m == 'task_mach2021-02-25/'):

            img_g = rgb2gray(img)
            val = filters.threshold_otsu(img_g)
            mask = img_g < val
            imlabel = skimage.measure.label(mask, background=0)
            print('objects: ', np.max(imlabel))
            props = skimage.measure.regionprops(imlabel)
            for j in range(0, len(props)):
                y0, x0, y1, x1 = props[j].bbox
                img_k = img[y0:y1, x0:x1]
                x, y, w = img_k.shape
                if x > 10 and y > 10 and x<50 and y < 50:
                    io.imsave(path_img_targer_false + name_i + '_false_s_' + str(j) + '.jpg', img_k)





#plt.imshow(img)
#plt.show()
