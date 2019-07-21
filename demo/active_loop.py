from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import numpy as np
import cv2
import random
import math
import subprocess
import argparse
import csv

from tqdm import tqdm
from shutil import copy
from active_loop_components import active_loop_iteration

if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="Open Images Evaluation")
    parser.add_argument(
        "--eval-category",
        default="Person",
        metavar="FILE",
        help="Test category",
    )
    parser.add_argument(
        "--input-feedback-dir",
        default=None,
        metavar="FILE",
        help="Directory to grab user feedback from",
    )
    parser.add_argument(
        "--request-feedback-dir",
        default=None,
        metavar="FILE",
        help="Directory to save data for user feedback",
    )

    args = parser.parse_args()

    image_dir = '/n/pana/scratch/ravi/open_images'
    sub_dirs = [ 'train_' + i for i in \
                  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

    image_paths = []
    for sd in sub_dirs:
        image_paths = image_paths + glob.glob(os.path.join(image_dir, sd, '*.jpg'))

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    class_name_file = os.path.join(image_dir, 'labels', 'class-descriptions-boxable.csv')

    class_name_to_id = {}
    with open(class_name_file) as class_names:
        reader = csv.reader(class_names, delimiter=',')
        for row in reader:
            class_name_to_id[row[1]] = row[0]

    assert(args.eval_category in class_name_to_id)
    class_id = class_name_to_id[args.eval_category]

    input_user_feedback = None
    if args.input_feedback_dir:
        input_user_feedback_path = os.path.join(args.input_feedback_dir, 'aggregate_user_feedback.npy')
        input_user_feedback = np.load('aggregate_user_feedback.npy')[()]

    classes_of_interest = [ (class_id, args.eval_category)]
    print(classes_of_interest)

    sample_dir = '/n/pana/scratch/ravi/open_images_sub_sample_all_long_tail/'
    sample_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))

    sample_id_to_path = {}
    for path in tqdm(sample_paths[:1000]):
        sample_id_to_path[path.split('/')[-1].split('.')[0]] = path

    images_by_class = np.load('openimages_images_by_class.npy')[()]
    for cls_id in images_by_class.keys():
        images_by_class[cls_id] = set(images_by_class[cls_id])

    cls_counts = {}
    search_set = set(sample_id_to_path.keys())

    for cls_id in images_by_class.keys():
        cls_counts[cls_id] = 0
        for im_id in images_by_class[cls_id]:
            if im_id in search_set:
                cls_counts[cls_id] = cls_counts[cls_id] + 1

    instances_by_class = np.load('openimages_instances_by_class.npy')[()]

    instances_by_img = {}
    for cls_id in instances_by_class.keys():
        for ins in instances_by_class[cls_id]:
            if ins['image_id'] not in instances_by_img:
                instances_by_img[ins['image_id']] = [ins]
            else:
                instances_by_img[ins['image_id']].append(ins)

    num_query_examples = 5
    query_instances = {}

    for cls_id, name in classes_of_interest:
        curr_query_examples = 0
        sourced_images = []
        for ins in instances_by_class[cls_id]:
            if ins['image_id'] in sourced_images:
                continue
            sourced_images.append(ins['image_id'])
            img = cv2.imread(image_id_to_path[ins['image_id']])
            h = img.shape[0]
            w = img.shape[1]
            x1 = int(float(ins['bbox'][0]) * w)
            x2 = int(float(ins['bbox'][1]) * w)
            y1 = int(float(ins['bbox'][2]) * h)
            y2 = int(float(ins['bbox'][3]) * h)
            if ((y2-y1) * (x2-x1) > 128 * 128):
                patch = img[y1:y2, x1:x2, :]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cls_name = name.replace(' ', '_')
                query_instances[(cls_id, curr_query_examples)] = (ins, patch)
                curr_query_examples = curr_query_examples + 1
                if curr_query_examples >= num_query_examples:
                    break

    model_name = 'resnet_v2_50'
    ranked_images, feedback_images, similarity_vis_images = \
        active_loop_iteration(query_instances,
                              num_query_examples,
                              {}, {}, {}, model_name,
                              sample_id_to_path)

    assert(args.request_feedback_dir)

    for cls_id, name in classes_of_interest:
        cls_output_dir = os.path.join(args.request_feedback_dir, name.replace(' ', '_'))
        if not os.path.exists(cls_output_dir):
            os.makedirs(cls_output_dir)

        feedback_dir =  os.path.join(cls_output_dir, 'feedback_images')
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)

        request_user_feedback_path = os.path.join(cls_output_dir, 'request_feedback.npy')
        ranked_images_path = os.path.join(cls_output_dir, 'ranked_images.npy')
        np.save(ranked_images_path, ranked_images[cls_id])
        np.save(request_user_feedback_path, feedback_images[cls_id])
        for img_id in feedback_images[cls_id]:
            copy(sample_id_to_path[img_id], feedback_dir)
