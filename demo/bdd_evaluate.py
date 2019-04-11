import os
import numpy as np
import json
import glob
from tqdm import tqdm
from collections import defaultdict

def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])

    return ap

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'name')
    image_gt_boxes = {k: np.array([[float(z) for z in b['bbox']]
                                   for b in boxes])
                      for k, boxes in image_gts.items()}
    image_gt_checked = {k: np.zeros((len(boxes), len(thresholds)))
                        for k, boxes in image_gts.items()}
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['bbox']
        ovmax = -np.inf
        jmax = -1
        try:
            gt_boxes = image_gt_boxes[p['name']]
            gt_checked = image_gt_checked[p['name']]
        except KeyError:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], box[0])
            iymin = np.maximum(gt_boxes[:, 1], box[1])
            ixmax = np.minimum(gt_boxes[:, 2], box[2])
            iymax = np.minimum(gt_boxes[:, 3], box[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((box[2] - box[0] + 1.) * (box[3] - box[1] + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(num_gts)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])

    return recalls, precisions, ap

def compute_map_on_list(label_list, predictions_path):
    gt_list = []
    pred_list = []
    classes = {}
    for l in tqdm(label_list):
        with open(l) as data_file:
            image_id = l.split('/')[-1].split('.')[0]
            data = json.load(data_file)
            cat_gt = group_by_key(data['frames'][0]['objects'], 'category')
            for cls in cat_gt.keys():
                if cls not in classes:
                    classes[cls] = []

        for b in data['frames'][0]['objects']:
            if 'box2d' in b:
                obj_dict= { 'bbox': [b['box2d']['x1'], b['box2d']['y1'],
                                     b['box2d']['x2'], b['box2d']['y2']],
                            'name': image_id,
                            'category': b['category']
                          }
                gt_list.append(obj_dict)

    with open(predictions_path) as pred_file:
        pred_list = json.load(pred_file)

    print(len(gt_list), len(pred_list))
    cat_gt = group_by_key(gt_list, 'category')
    cat_pred = group_by_key(pred_list, 'category')
    print('gt_classes', cat_gt.keys())
    print('pred_classes', cat_pred.keys())

    cat_list = sorted(cat_gt.keys())
    thresholds = list(np.linspace(.5, 0.95, np.round((0.95 - .5) / .05) + 1, endpoint=True))
    print(thresholds)
    aps = np.zeros((len(thresholds), len(cat_list)))
    for i, cat in enumerate(cat_list):
        if cat in cat_pred:
            r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
            aps[:, i] = ap
            print(cat, np.mean(ap * 100), len(cat_gt[cat]))
    aps *= 100
    mAP = np.mean(aps)
    print('Final', mAP)

def get_attribute_slices(bdd_labels_list):
    attribute_slices = {}
    for l in tqdm(bdd_labels_list):
        with open(l) as data_file:
            data = json.load(data_file)
            for attr in data['attributes'].keys():
                attr_val = data['attributes'][attr]
                if attr_val not in attribute_slices:
                    attribute_slices[attr_val] = []
                    attribute_slices[attr_val].append(l)
                else:
                    attribute_slices[attr_val].append(l)
    return attribute_slices


bdd_labels_path = '/n/pana/scratch/ravi/bdd/bdd100k/labels/100k/val'
bdd_labels_list = glob.glob(os.path.join(bdd_labels_path, '*.json'))

#attribute_slices = get_attribute_slices(bdd_labels_list)

#for attr_val in attribute_slices.keys():
#    print(attr_val, len(attribute_slices[attr_val]))
#    compute_map_on_list(attribute_slices[attr_val],
#                        'supervised.json')
compute_map_on_list(bdd_labels_list, 'supervised_10.json')
