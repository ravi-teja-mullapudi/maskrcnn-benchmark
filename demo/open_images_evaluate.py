import os
import numpy as np
import json
import glob
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sample_open_images import get_instances_per_class
import cv2
import argparse

def draw_boxes(img, labels):
    img = img.copy()
    for l in labels:
        bbox = np.array(l['bbox']).astype(int)
        top_left, bottom_right = bbox[:2].tolist(), bbox[2:].tolist()
        img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right),
                            (0, 255, 0), 3)
        cv2.putText(img, l['category'], tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 0, 0), 2)
    return img

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

def compute_map_on_list(gt_list, pred_list):
    print(len(gt_list), len(pred_list))
    cat_gt = group_by_key(gt_list, 'category')
    cat_pred = group_by_key(pred_list, 'category')
    print('gt_classes', cat_gt.keys())
    print('pred_classes', cat_pred.keys())

    print(gt_list[0], pred_list[0])
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Images Evaluation")
    parser.add_argument(
        "--split",
        default='validation',
        metavar="FILE",
        help="Choose split to evaluate on",
    )
    parser.add_argument(
        "--predictions",
        default=None,
        metavar="FILE",
        help="path to predictions file",
    )
    parser.add_argument(
        "--eval-category",
        default="Person",
        metavar="FILE",
        help="Test category",
    )
    parser.add_argument(
        "--output-dir",
        default="./",
        help="Output directory for visualiation images",
    )

    args = parser.parse_args()

    image_dir = '/n/pana/scratch/ravi/open_images/'
    image_paths = []
    bbox_file = None
    if args.split == 'train':
        sub_dirs = [ 'train_' + i for i in \
                        [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

        for sd in sub_dirs:
            image_paths = image_paths + glob.glob(os.path.join(image_dir, sd, '*.jpg'))
        bbox_file = os.path.join(image_dir, 'labels', 'train-annotations-bbox.csv')
    elif args.split == 'validation':
        image_paths = glob.glob(os.path.join(image_dir, 'validation', '*.jpg'))
        bbox_file = os.path.join(image_dir, 'labels', 'validation-annotations-bbox.csv')

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    class_name_file = os.path.join(image_dir, 'labels', 'class-descriptions-boxable.csv')

    instances_class_id, class_id_to_name = \
            get_instances_per_class(class_name_file, bbox_file)
    images_class_id = {}

    cat_names = set()
    for cls in instances_class_id.keys():
        for ins in instances_class_id[cls]:
            if cls in images_class_id:
                images_class_id[cls].add(ins['image_id'])
            else:
                images_class_id[cls] = set([ins['image_id']])

            if ins['category'] not in cat_names:
                cat_names.add(ins['category'])

    class_name_to_id = { 'House': '/m/03jm5',
                         'Fire hydrant':  '/m/01pns0',
                         'Limousine': '/m/01lcw4',
                         'Billboard': '/m/01knjb',
                         'Person': '/m/01g317',
                         'Motorcycle': '/m/04_sv',
                         'Parking meter': '/m/015qbp',
                         'Ambulance': '/m/012n7d',
                         'Bicycle': '/m/0199g',
                         'Bus': '/m/01bjv',
                         'Convenience store': '/m/0crjs',
                       }

    assert(args.eval_category in class_name_to_id)
    class_id = class_name_to_id[args.eval_category]

    cls_instances_per_img = group_by_key(instances_class_id[class_id], 'image_id')

    with open(args.predictions) as pred_file:
        pred_list = json.load(pred_file)

    pred_per_img = group_by_key(pred_list, 'name')

    gt_instances_cls = {}

    for img_id in tqdm(cls_instances_per_img.keys()):
        for ins in cls_instances_per_img[img_id]:
            img_path = image_id_to_path[ins['image_id']]
            img = Image.open(img_path).convert('RGB')
            width, height = img.size

            scaled_box = [  float(ins['bbox'][0]) * width,
                            float(ins['bbox'][2]) * height,
                            float(ins['bbox'][1]) * width,
                            float(ins['bbox'][3]) * height
                        ]

            cls = ins['category']
            gt_instance = { 'category' : class_id_to_name[cls],
                            'bbox' : scaled_box,
                            'name' : ins['image_id']
                          }

            if cls not in gt_instances_cls:
                gt_instances_cls[cls] = [gt_instance]
            else:
                gt_instances_cls[cls].append(gt_instance)

    gt_per_img = group_by_key(gt_instances_cls[class_id], 'name')
    predictions_cls = []

    for im_id in sorted(gt_per_img.keys()):
        if im_id in pred_per_img:
            predictions_cls = predictions_cls + pred_per_img[im_id]

    count = 0
    for im_id in sorted(gt_per_img.keys()):
        i = image_id_to_path[im_id]
        img = cv2.imread(i)
        if im_id in pred_per_img:
            pred_img = draw_boxes(img, pred_per_img[im_id])
        else:
            pred_img = img
        gt_img = draw_boxes(img, gt_per_img[im_id])
        vis_img = np.concatenate((pred_img, gt_img), axis=1)
        out_path = os.path.join(args.output_dir, 'open_images_test_' + im_id + '.png')
        cv2.imwrite(out_path, vis_img)
        if count > 100:
            break
        count = count + 1

    compute_map_on_list(gt_instances_cls[class_id], predictions_cls)
