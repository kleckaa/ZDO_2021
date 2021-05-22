import os
import glob

import skimage
from skimage import filters
import joblib
from skimage import io
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn import neighbors
from skimage.measure import label

pos_im_path = 'anot/True/'
neg_im_path = 'anot/False/'

pos_im_listing = os.listdir(pos_im_path)
neg_im_listing = os.listdir(neg_im_path)

num_pos_samples = len(pos_im_listing)
num_neg_samples = len(neg_im_listing)

print(f'pozitivní data trénovací množiny: {num_pos_samples}')
print(f'negativní data trénovací množiny: {num_neg_samples}')

data_train = []
labels_train = []
l_x = []
l_y = []

radius = 1
n_points = 8 * radius
METHOD = 'uniform'
n_bins = 10

for file in pos_im_listing:
    img = io.imread(pos_im_path + file)
    gray_img = rgb2gray(img)

    x, y = gray_img.shape
    '''
    if y>x:
        gray_img = rotate(gray_img, 90)
        x, y = gray_img.shape
    '''
    if x >10 and y > 10:

        l_x.append(x)
        l_y.append(y)
        '''
        image_resized = resize(gray_img, (10,10))
        #image_resized = gray_img

        lbp = local_binary_pattern(image_resized, n_points, radius, METHOD)

        #n_bins = int(lbp.max() + 1)
        #print(n_bins)
        a, b, c = plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
        '''
        r = img[round(x / 2), round(y / 2), 0]
        g = img[round(x / 2), round(y / 2), 1]
        b = img[round(x / 2), round(y / 2), 2]

        val = filters.threshold_otsu(gray_img)
        mask = gray_img < val
        imlabel = skimage.measure.label(mask, background=0)
        props = skimage.measure.regionprops(imlabel)

        convex = []
        area = []
        perimeter = []
        for kj in range(0,len(props)):
            convex.append(props[kj].convex_area)
            area.append(props[kj].area)
            perimeter.append(props[kj].perimeter)

        r = []
        b = []
        g = []
        #print(mask)
        # kx, ky = mask.shape
        for kl in range(0, x):
            for km in range(0, y):
                if imlabel[kl, km]:
                    r.append(img[round(x / 2), round(y / 2), 0])
                    g.append(img[round(x / 2), round(y / 2), 1])
                    b.append(img[round(x / 2), round(y / 2), 2])

        r_color = sum(r) / len(r)
        g_color = sum(g) / len(g)
        b_color = sum(b) / len(b)
        aa = (r_color, g_color, b_color, sum(area), sum(perimeter), sum(convex))

        data_train.append(aa)
        labels_train.append(1)
        #plt.show()

print('nejmenší výška obrázku:' + str(min(l_x)))
print('nejmenší šířka obrázku:' + str(min(l_y)))
print('největší výška obrázku:' + str(max(l_x)))
print('největší šířka obrázku:' + str(max(l_y)))

print(str(len(gray_img)))
#print(lbp)
#plt.imshow(image_resized)
#plt.show()

for file in neg_im_listing:
    img = io.imread(neg_im_path + file)
    gray_img = rgb2gray(img)
    x, y = gray_img.shape
    if x > 10 and y > 10:
        '''
        image_resized = resize(gray_img, (10, 10))
        #image_resized = gray_img
        lbp = local_binary_pattern(image_resized, n_points, radius, METHOD)
        #n_bins = int(lbp.max() + 1)
        a, b, c = plt.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins))
        '''

        val = filters.threshold_otsu(gray_img)
        mask = gray_img < val
        imlabel = skimage.measure.label(mask, background=0)
        props = skimage.measure.regionprops(imlabel)

        convex = []
        area = []
        perimeter = []
        for kj in range(0, len(props)):
            convex.append(props[kj].convex_area)
            area.append(props[kj].area)
            perimeter.append(props[kj].perimeter)

        r = []
        b = []
        g = []
        # print(mask)
        # kx, ky = mask.shape
        for kl in range(0, x):
            for km in range(0, y):
                if imlabel[kl, km]:
                    r.append(img[round(x / 2), round(y / 2), 0])
                    g.append(img[round(x / 2), round(y / 2), 1])
                    b.append(img[round(x / 2), round(y / 2), 2])

        r_color = sum(r) / len(r)
        g_color = sum(g) / len(g)
        b_color = sum(b) / len(b)
        aa = (r_color, g_color, b_color, sum(area), sum(perimeter), sum(convex))

        data_train.append(aa)
        labels_train.append(0)

le = LabelEncoder()
labels = le.fit_transform(labels_train)

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(data_train), labels_train, test_size=0.10, random_state=32)
print('done')
model = LinearSVC(C = 0.9, tol = 1e-4)
model.fit(trainData, trainLabels)
print('SVM natrénovaný')
predictions = model.predict(testData)
print(classification_report(testLabels, predictions))


print('__________')
print('__________')


knn=neighbors.KNeighborsClassifier(n_neighbors = 5)
knn.fit(trainData, trainLabels)
print('KNN natrénovaný')
predictions = knn.predict(testData)
print(classification_report(testLabels, predictions))

joblib.dump(model, 'SVM_s.npy')
joblib.dump(knn, 'KNN_s.npy')
