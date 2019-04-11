import os
import numpy as np
import json
import glob
import cv2

from tqdm import tqdm
from collections import defaultdict
from bdd_analysis import group_by_key, cat_pc
from bdd_visualize import draw_boxes

def get_unmatched_objects(label_file, pred_all, unmatched_objects = {},
                          gt_large_objects={}, small_object_thresh = 32 * 32,
                          iou_thresh = 0.5):
    gt_list = []
    with open(l) as data_file:
        image_id = l.split('/')[-1].split('.')[0]
        data = json.load(data_file)

    unmatched_objects[image_id] = []
    gt_large_objects[image_id] = []

    for b in data['frames'][0]['objects']:
            if 'box2d' in b:
                x1 = b['box2d']['x1']
                y1 = b['box2d']['y1']
                x2 = b['box2d']['x2']
                y2 = b['box2d']['y2']
                if (x2 - x1) * (y2 - y1) > small_object_thresh:
                    obj_dict= { 'bbox': [x1, y1, x2, y2],
                                'name': image_id,
                                'category': b['category']
                              }
                    gt_list.append(obj_dict)
                    gt_large_objects[image_id].append(obj_dict)

    pred_list = []
    for pred in pred_all:
        if pred['name'] == image_id and pred['score'] > 0.5:
            pred_list.append(pred)

    #print(len(gt_list), len(pred_list))
    cat_gt = group_by_key(gt_list, 'category')
    cat_pred = group_by_key(pred_list, 'category')
    #print('gt_classes', cat_gt.keys())
    #print('pred_classes', cat_pred.keys())

    cat_list = sorted(cat_gt.keys())
    thresholds = [iou_thresh]

    for i, cat in enumerate(cat_list):
        if cat in cat_gt:
            _, _, _, image_gts, gt_matched, predictions, fp  = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
            if gt_matched is not None:
                gt_not_matched = [ not bool(v[0]) for v in gt_matched ]
                for d in np.array(image_gts[image_id])[gt_not_matched]:
                    unmatched_objects[image_id].append(d)
            else:
                for d in image_gts[image_id]:
                    unmatched_objects[image_id].append(d)

if __name__ == "__main__":
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    bdd_labels_path = '/n/pana/scratch/ravi/bdd/bdd100k/labels/100k/val'
    bdd_labels_list = glob.glob(os.path.join(bdd_labels_path, '*.json'))
    output_dir = '/n/pana/scratch/ravi/bdd_missed_large_objects'

    with open('supervised.json') as pred_file:
        pred_all = json.load(pred_file)

    unmatched_objects = {}
    gt_large_objects = {}

    for l in tqdm(bdd_labels_list):
        get_unmatched_objects(l, pred_all, unmatched_objects, gt_large_objects)
        image_id = l.split('/')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_id + '.jpg')
        img = cv2.imread(image_path)
        unmatched = draw_boxes(img, unmatched_objects[image_id])

        pred_list = []
        for p in pred_all:
            if p['name'] == image_id and p['score'] > 0.5:
                pred_list.append(p)

        #predicted = draw_boxes(img, pred_list)
        #vis_img = np.concatenate((unmatched, predicted), axis=1)
        #if len(unmatched) > 0:
        #    cv2.imwrite(os.path.join(output_dir, image_id + '.png'), vis_img)

    all_unmatched = sum(unmatched_objects.values(), [])
    all_large_objects = sum(gt_large_objects.values(), [])
    cat_large_gt = group_by_key(all_large_objects, 'category')
    cat_unmatched = group_by_key(all_unmatched, 'category')

    for cat in cat_large_gt.keys():
        print(cat, len(cat_unmatched[cat]), len(cat_large_gt[cat]))
    #total_missed = sum([ len(unmatched_objects[k]) for k in unmatched_objects.keys() ] )
    #print(total_missed)
