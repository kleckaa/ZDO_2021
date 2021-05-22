import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import filters
from skimage.measure import label
import skimage

import joblib
from skimage.transform import rotate
from skimage.util import crop
from skimage.feature import local_binary_pattern
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.draw import rectangle_perimeter, set_color

def predict(img_path, SVM_model, threshold):

    pre = []

    img_main = io.imread(img_path)
    image = rgb2gray(img_main)

    val = filters.threshold_otsu(image)
    mask = image < val
    imlabel = skimage.measure.label(mask, background=0)
    props = skimage.measure.regionprops(imlabel)

    model = joblib.load(SVM_model)

    radius = 1
    n_points = 8 * radius
    METHOD = 'uniform'
    n_bins = 10



    for i in range(0, len(props)):
        y0, x0, y1, x1 = props[i].bbox
        #img = image[y0:y1, x0:x1]
        img = img_main[y0:y1, x0:x1]
        img_g = image[y0:y1, x0:x1]
        img_crop = img.copy()
        x, y, h = img_crop.shape

        if x > 10 and y > 10 and x < 60 and y < 60:
            # print('y')
            '''
            if y>x: 
                img_crop = skimage.transform.rotate(img_crop,angle=90,resize=True, mode='constant')
                x, y = img_crop.shape
            '''

            #image_resized = resize(img_crop, (10,10))
            '''
            image_resized = resize(img_g, (10, 10))
            lbp = local_binary_pattern(image_resized, n_points, radius, METHOD)
            a, b, c = plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
            #plt.close()
            # print(a)
            '''

            val_img = filters.threshold_otsu(img_g)
            mask_img = img_g < val_img
            val_imlabel = skimage.measure.label(mask_img, background=0)
            props_img = skimage.measure.regionprops(val_imlabel)

            convex = []
            area = []
            perimeter = []
            for kj in range(0, len(props_img)):
                convex.append(props_img[kj].convex_area)
                area.append(props_img[kj].area)
                perimeter.append(props_img[kj].perimeter)

            r = []
            b = []
            g = []
            # print(mask)
            # kx, ky = mask.shape
            for kl in range(0, x):
                for km in range(0, y):
                    if val_imlabel[kl, km]:
                        r.append(img[round(x / 2), round(y / 2), 0])
                        g.append(img[round(x / 2), round(y / 2), 1])
                        b.append(img[round(x / 2), round(y / 2), 2])

            r_color = sum(r) / len(r)
            g_color = sum(g) / len(g)
            b_color = sum(b) / len(b)
            aa = (r_color, g_color, b_color, sum(area), sum(perimeter), sum(convex))

            aa = np.array(aa)
            aa = aa.reshape(1, -1)
            pred = model.predict(aa)
            if pred == 1:
                #if model.decision_function(a) > threshold:
                    #print(str(y0) + ' ' + str(x0))
                start = (y0, x0)
                end = (y1, x1)

                rr, cc = rectangle_perimeter(start, end=end, shape=img_main.shape)
                set_color(img_main, (rr, cc), (255, 0, 0))

                #pre.append([y0,x0,y1,x1,start,end,model.decision_function(a)])
                pre.append([y0, x0, y1, x1, start, end])
    plt.close()
    return pre,img_main






