import os
import skimage.io
import glob
import numpy as np
from pathlib import Path
import prediction as pre
from sklearn.metrics import f1_score
# def test_run_random():

def detection_score(gt_ann, prediction, iou_t):
    TP = 1
    all_ground = len(gt_ann)
    all_det = len(prediction)
    for i in range(0, len(gt_ann)):
        xmin_gt = gt_ann[i][1]
        ymin_gt = gt_ann[i][0]
        xmax_gt = gt_ann[i][3]
        ymax_gt = gt_ann[i][2]
        for j in range(0, len(prediction)):
            xmin_p = prediction[j][1]
            ymin_p = prediction[j][0]
            xmax_p = prediction[j][3]
            ymax_p = prediction[j][2]

            xmin = max(xmin_gt, xmin_p)
            ymin = max(ymin_gt, ymin_p)
            xmax = min(xmax_gt, xmax_p)
            ymax = min(ymax_gt, ymax_p)

            interArea = abs(max((xmax - xmin, 0)) * max((ymax - ymin), 0))
            boxAArea = abs((xmax_gt - xmin_gt) * (ymax_gt - ymin_gt))
            boxBArea = abs((xmax_p - xmin_p) * (ymax_p - ymin_p))
            iou = interArea / float(boxAArea + boxBArea - interArea)
            #print(iou)

            if iou >= iou_t:
                TP = TP + 1

    Precision = (TP / all_det)
    Recall = (TP / all_ground)
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    return Precision, Recall, F1


def prepare_ground_true_masks(gt_ann, filname):
    ground_true = []

    for num in range(0, len(gt_ann['images'])):
        if gt_ann['images'][num]['file_name'] == filname:
            img_id = gt_ann['images'][num]['id']

    for i in range(0, len(gt_ann['annotations'])):
        if gt_ann['annotations'][i]['image_id'] == img_id:
            found = False

            #print(gt_ann['annotations'][i]['bbox'])
            #print(gt_ann['annotations'][i]['area'])
            x0, y0, width, height = gt_ann['annotations'][i]['bbox']
            y0 = round(y0)
            x0 = round(x0)
            width = round(width)
            height = round(height)

            start = (y0, x0)
            end = (y0 + height, x0 + width)

            ground_true.append([y0, x0, y0 + height, x0 + width, start, end])

    return ground_true





#'test/images\\Original_1323_image.jpg'




# gt_ann = json.loads(Path(dataset_path)/"annotations/instances_default.json")
# assert f1score(ground_true_masks, prediction) > 0.55



