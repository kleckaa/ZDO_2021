import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage import filters
from skimage.measure import label
import skimage
import joblib

class VarroaDetector():
    def __init__(self):
        pass

    def predict(self, data):
        """
        :param data: np.ndarray with shape [pocet_obrazku, vyska, sirka, barevne_kanaly]
        :return: shape [pocet_obrazku, vyska, sirka], 0 - nic, 1 - varroa destructor
        """
        model = 'model/KNN_s.npy'
        print(data.shape[0])
        arr_mask_comp = []
        for num_first_dim in range(0, data.shape[0]):
            main_image = data[num_first_dim, :, :, :]
            arr_mask = np.zeros(main_image.shape[:2])
            image_gray = rgb2gray(main_image)

            val = filters.threshold_otsu(image_gray)
            mask = image_gray < val
            imlabel = skimage.measure.label(mask, background=0)
            props = skimage.measure.regionprops(imlabel)

            model = joblib.load(model)

            for i in range(0, len(props)):
                y0, x0, y1, x1 = props[i].bbox
                img_obj = main_image[y0:y1, x0:x1]
                img_g = image_gray[y0:y1, x0:x1]
                x, y, h = img_obj.shape

                if x > 10 and y > 10 and x < 60 and y < 60:

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
                                r.append(img_obj[round(x / 2), round(y / 2), 0])
                                g.append(img_obj[round(x / 2), round(y / 2), 1])
                                b.append(img_obj[round(x / 2), round(y / 2), 2])

                    r_color = sum(r) / len(r)
                    g_color = sum(g) / len(g)
                    b_color = sum(b) / len(b)
                    aa = (r_color, g_color, b_color, sum(area), sum(perimeter), sum(convex))

                    aa = np.array(aa)
                    aa = aa.reshape(1, -1)
                    pred = model.predict(aa)
                    if pred == 1:
                        yy0, xx0, yy1, xx1 = props_img[kj].bbox
                        print(props_img[kj])
                        arr_mask[y0:y1, x0:x1] = imlabel[y0:y1, x0:x1]

            arr_mask_comp.append(arr_mask)

        fin_arr = np.array(arr_mask_comp)
        return fin_arr