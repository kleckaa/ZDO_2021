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

Pr = []
Re = []
F1_a = []

for fc in range(0,len(files)):
    filename = files[fc]
    im = skimage.io.imread(filename)



    #class_model = 'model/SVM_s.npy' #Cesta k SVM modelu klasifikátoru
    class_model = 'model/KNN_s.npy' # Cesta k KNN modelu klasifikátoru
    threshold = 1
    prediction, img_predict = pre.predict(filename, class_model, threshold) # Metoda, která na provede predikci
    assert img_predict.shape[0] == im.shape[0]


    #plt.imshow(img_predict) # Vykreslení predikce klestiku
    #plt.show()


    with open(dataset_path + '/annotations/instances_default.json', "r") as read_file:
        gt_ann = json.load(read_file)
    filname = filename.replace('test/images\\','')
    gt = tst.prepare_ground_true_masks(gt_ann, filname) # nacteni ground truth boxů
    Precision, Recall, F1 = tst.detection_score(gt, prediction, 0.3) # metoda, která počítá precision, recall, f1 score

    print(filename)
    print('Precision -> ' + str(Precision))
    print('Recall -> ' + str(Recall))
    print('F1 -> '+ str(F1))
    print( '________')

    Pr.append(Precision)
    Re.append(Recall)
    F1_a.append(F1)

print('Průměr:')
print('Precision -> ' +  str(sum(Pr)/len(files)))
print('Recall -> ' +  str(sum(Re)/len(files)))
print('F1 -> ' +  str(sum(F1_a)/len(files)))