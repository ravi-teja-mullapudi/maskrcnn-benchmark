import os
import numpy as np
import json
import glob
from tqdm import tqdm

category_id_to_name = { 10 : 'traffic light',
                        3 : 'car',
                        8 : 'truck',
                        6 : 'bus',
                        7 : 'train',
                        4 : 'motor',
                        1 : 'person',
                        2 : 'bike',
                        13 : 'traffic sign' }

maskrcnn_labels_path = '/n/pana/scratch/ravi/bdd/bdd100k/maskrcnn_labels/100k/train/predictions_100k.npy'
maskrcnn_labels = np.load(maskrcnn_labels_path)[()]

image_ids = []
classes = []
annotations = []

for l in tqdm(maskrcnn_labels.keys()):
    image_id = l.split('/')[-1].split('.')[0]
    pred_boxes, pred_cls, pred_scores, _ = maskrcnn_labels[l]
    num_objects = pred_scores.shape[0]

    height = 720
    width = 1280
    detection_thresh = 0.5

    for b in range(num_objects):
        y1 = pred_boxes[b][0] * height
        x1 = pred_boxes[b][1] * width
        y2 = pred_boxes[b][2] * height
        x2 = pred_boxes[b][3] * width

        if pred_cls[b] in category_id_to_name and \
           pred_scores[b] > detection_thresh:

            if image_id not in image_ids:
                image_ids.append(image_id)

            obj_dict = { 'name': image_id,
                         'category': category_id_to_name[pred_cls[b]],
                         'bbox': [x1, y1, x2, y2]
                       }
            annotations.append(obj_dict)

            if obj_dict['category'] not in classes:
                classes.append(obj_dict['category'])

bdd_annotations = { 'image_ids' : image_ids,
                    'classes' : classes,
                    'annotations' : annotations
                  }
print(classes)
with open('bdd_maskrcnn_train.json', 'w') as fp:
    json.dump(bdd_annotations, fp)
