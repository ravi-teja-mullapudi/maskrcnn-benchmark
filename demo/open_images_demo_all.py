# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from open_images_predictor import OpenImagesDemo
from sample_open_images import get_instances_per_class

import os
import glob
import time
import json
import csv
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/train_open_images_1k_instances_config.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--predictions-out",
        default="./test.json",
        metavar="FILE",
        help="path to file to output labels",
    )
    parser.add_argument(
        "--image-dir",
        default="/n/pana/scratch/ravi/open_images/",
        metavar="FILE",
        help="path to test image directory",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "--show-mask-heatmaps",
        dest="show_mask_heatmaps",
        help="Show a heatmap probability for the top masks-per-dim masks",
        action="store_true",
    )
    parser.add_argument(
        "--masks-per-dim",
        type=int,
        default=2,
        help="Number of heatmaps per dimension to show",
    )
    parser.add_argument(
        "--train-data",
        default=None,
        metavar="FILE",
        help="Test category",
    )
    parser.add_argument(
        "--split",
        default='validation',
        metavar="FILE",
        help="Choose split to evaluate on",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    classes_of_interest = []
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

    with open(args.train_data) as data_file:
        data = json.load(data_file)
    categories = data['classes']

    for c in categories:
        assert(c in class_name_to_id)
        classes_of_interest.append((class_name_to_id[c], c))

    # prepare object that handles inference plus adds predictions on top of image
    open_images_demo = OpenImagesDemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
        categories = categories
    )

    image_paths = []
    bbox_file = None
    if args.split == 'train':
        sub_dirs = [ 'train_' + i for i in \
                [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

        image_paths = []
        for sd in sub_dirs:
            image_paths = image_paths + glob.glob(os.path.join(args.image_dir, sd, '*.jpg'))
        bbox_file = os.path.join(image_dir, 'labels', 'train-annotations-bbox.csv')
    elif args.split == 'validation':
        image_paths = glob.glob(os.path.join(args.image_dir, 'validation', '*.jpg'))
        bbox_file = os.path.join(args.image_dir, 'labels', 'validation-annotations-bbox.csv')

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    class_name_file = os.path.join(args.image_dir, 'labels', 'class-descriptions-boxable.csv')

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

    pred_list = []

    inference_image_ids = []
    for cls in instances_class_id.keys():
        inference_image_ids = inference_image_ids + list(images_class_id[cls])

    for img_id in tqdm(sorted(image_id_to_path.keys())[:25000]):
        i = image_id_to_path[img_id]
        img = cv2.imread(i)
        image_id = i.split('/')[-1].split('.')[0]

        start = time.time()
        predictions = open_images_demo.compute_prediction(img)
        end = time.time()
        scores = predictions.get_field('scores')

        high_conf_idx = scores > args.confidence_threshold
        predictions = predictions[high_conf_idx]
        scores = scores[high_conf_idx]

        boxes = predictions.bbox
        labels = predictions.get_field('labels')
        labels = [ open_images_demo.categories[l] for l in labels ]

        for b in range(len(labels)):
            obj_dict = { 'name' : image_id,
                         'bbox' : [float(boxes[b][0]),
                                   float(boxes[b][1]),
                                   float(boxes[b][2]),
                                   float(boxes[b][3])],
                         'category' : labels[b],
                         'score' : float(scores[b])
                        }
            pred_list.append(obj_dict)

    with open(args.predictions_out, 'w') as fp:
        json.dump(pred_list, fp)

if __name__ == "__main__":
    main()
