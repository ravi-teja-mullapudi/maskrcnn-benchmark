# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import os
import glob
import time
import json
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="../configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
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
        "--test-image-dir",
        default="/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/",
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

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_threshold=args.confidence_threshold,
        show_mask_heatmaps=args.show_mask_heatmaps,
        masks_per_dim=args.masks_per_dim,
        min_image_size=args.min_image_size,
    )

    image_paths = glob.glob(os.path.join(args.test_image_dir, '*.jpg'))
    pred_list = []
    coco_cat_to_bdd_cat = { "person" : "person",
                            "car" : "car",
                            "traffic light" : "traffic light",
                            "stop sign" : "traffic sign",
                            "bus" : "bus",
                            "truck" : "truck",
                            "bicycle" : "bike",
                            "motorcycle" : "motor",
                            "train" : "train" }

    for i in tqdm(image_paths):
        img = cv2.imread(i)
        image_id = i.split('/')[-1].split('.')[0]

        start = time.time()
        predictions = coco_demo.compute_prediction(img)
        end = time.time()
        scores = predictions.get_field('scores')
        #high_conf_idx = scores > args.confidence_threshold
        #predictions = predictions[high_conf_idx]
        #scores = predictions.get_field('scores')
        boxes = predictions.bbox
        labels = predictions.get_field('labels')
        labels = [ coco_demo.CATEGORIES[l] for l in labels ]

        for b in range(len(labels)):
            if labels[b] in coco_cat_to_bdd_cat:
                label = coco_cat_to_bdd_cat[labels[b]]
                obj_dict = { 'name' : image_id,
                             'bbox' : [float(boxes[b][0]),
                                       float(boxes[b][1]),
                                       float(boxes[b][2]),
                                       float(boxes[b][3])],
                             'category' : label,
                             'score' : float(scores[b])
                            }
                pred_list.append(obj_dict)

    with open(args.predictions_out, 'w') as fp:
        json.dump(pred_list, fp)

if __name__ == "__main__":
    main()
