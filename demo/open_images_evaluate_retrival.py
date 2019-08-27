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

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

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

    cat_names = set()
    for cls in instances_class_id.keys():
        for ins in instances_class_id[cls]:
            if cls in images_class_id:
                images_class_id[cls].add(ins['image_id'])
            else:
                images_class_id[cls] = set([ins['image_id']])

            if ins['category'] not in cat_names:
                cat_names.add(ins['category'])

    with open(args.predictions) as pred_file:
        pred_list = json.load(pred_file)

    pred_per_img = group_by_key(pred_list, 'name')
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    score_per_img = {}
    for img_id in tqdm(sorted(image_id_to_path.keys())[:25000]):
        has_target_instance = False
        max_score = 0
        if img_id in pred_per_img:
            for ins in pred_per_img[img_id]:
                if ins['category'] == args.eval_category:
                    has_target_instance = True
                    max_score = max(max_score, ins['score'])

        if has_target_instance:
            if img_id in images_class_id[class_id]:
                tp = tp + 1
                score_per_img[img_id] = (max_score, 'tp')
            else:
                fp = fp + 1
                score_per_img[img_id] = (max_score, 'fp')
        else:
            if img_id in images_class_id[class_id]:
                fn = fn + 1
                score_per_img[img_id] = (max_score, 'fn')
            else:
                tn = tn + 1
                score_per_img[img_id] = (max_score, 'tn')

    print(tp, fp, fn, tn)
    score_sorted = sorted(score_per_img.items(), key=lambda i: -i[1][0])

    print(instances_class_id[class_id])

    cls_instances_per_img = group_by_key(instances_class_id[class_id], 'image_id')
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

    count = 0
    for im_id, s in score_sorted:
        i = image_id_to_path[im_id]
        img = cv2.imread(i)
        if s[1] != 'fp':
            continue
        if im_id in pred_per_img:
            pred_img = draw_boxes(img, pred_per_img[im_id])
        else:
            pred_img = img
        if im_id in gt_per_img:
            gt_img = draw_boxes(img, gt_per_img[im_id])
        else:
            gt_img = img
        vis_img = np.concatenate((pred_img, gt_img), axis=1)
        out_path = os.path.join(args.output_dir, 'false_pos_' + str(count) + '.png')
        cv2.imwrite(out_path, vis_img)
        if count > 100:
            break
        count = count + 1

    tp_top_25 = sum([ t[1][1] == 'tp' for t in score_sorted[:25] ])
    print('tp_top_25', tp_top_25)
    tp_top_100 = sum([ t[1][1] == 'tp' for t in score_sorted[:100] ])
    print('tp_top_100', tp_top_100)
