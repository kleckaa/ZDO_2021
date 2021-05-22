import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import prediction as pre
import test as tst
import json

dataset_path = 'test'
files = glob.glob(dataset_path + '/images/*.jpg')

num = np.random.randint(0, len(files)) #nahodny obrazek z restovaciho datasetu

filename = files[num]
im = skimage.io.imread(filename)

# class_model = 'model/SVM_s.npy' #Cesta k SVM modelu klasifikátoru
class_model = 'model/KNN_s.npy'  # Cesta k KNN modelu klasifikátoru
threshold = 1
prediction, img_predict = pre.predict(filename, class_model, threshold)  # Metoda, která na provede predikci

assert img_predict.shape[0] == im.shape[0]

with open('test/annotations/instances_default.json', "r") as read_file:
    gt_ann = json.load(read_file)
filname = filename.replace('test/images\\', '')
gt = tst.prepare_ground_true_masks(gt_ann, filname)  # nacteni ground truth boxů
Precision, Recall, F1 = tst.detection_score(gt, prediction, 0.3)  # metoda, která počítá precision, recall, f1 score

assert F1 > 0.01

print(filename)
print('Precision -> ' + str(Precision))
print('Recall -> ' + str(Recall))
print('F1 -> ' + str(F1))
print('________')

plt.imshow(img_predict) # Vykreslení predikce klestiku
plt.show()