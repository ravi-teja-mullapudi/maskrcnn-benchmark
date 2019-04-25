from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import numpy as np
import tensorflow as tf
import time
import cv2
import random

try:
    import urllib2 as urllib
except ImportError:
    import urllib.request as urllib

from tqdm import tqdm
from collections import defaultdict

from tensorflow.contrib import slim
sys.path.append(os.path.realpath('/n/pana/scratch/ravi/models/research/slim/'))

from nets import resnet_v2
from preprocessing import inception_preprocessing
from datasets import imagenet

def get_embeddings(instances):
    image_size = 299
    names = imagenet.create_readable_names_for_imagenet_labels()

    with tf.Graph().as_default():
        image = tf.placeholder(tf.uint8, (None, None, 3))
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_image = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, _ = resnet_v2.resnet_v2_101(processed_image, 1001, is_training=False)
            pool5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_101/pool5:0")

        init_fn = slim.assign_from_checkpoint_fn('resnet_v2_101.ckpt',
                                slim.get_model_variables('resnet_v2'))

        with tf.Session() as sess:
            init_fn(sess)

            for im_id in tqdm(instances.keys()):
                img = cv2.imread(instances[im_id]['path'])
                embedding_list = []
                for b in instances[im_id]['detections']:
                    x1 = int(b['bbox'][0])
                    x2 = int(b['bbox'][2])
                    y1 = int(b['bbox'][1])
                    y2 = int(b['bbox'][3])

                    if y2-y1 >= 8 and x2-x1 >= 8:
                        patch = img[y1:y2, x1:x2, :]
                        scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], feed_dict={image: patch})
                        b['pool5_resnet_v2_101'] = embedding[0, 0, 0, :]

def bdd_gt_embeddings(split):
    bdd_labels_path = os.path.join('/n/pana/scratch/ravi/bdd/bdd100k/labels/100k/', split)
    bdd_labels_list = glob.glob(os.path.join(bdd_labels_path, '*.json'))
    image_dir = os.path.join('/n/pana/scratch/ravi/bdd/bdd100k/images/100k/', split)

    instances = {}

    for l in tqdm(bdd_labels_list):
        image_id = l.split('/')[-1].split('.')[0]
        image_path = os.path.join(image_dir, image_id + '.jpg')

        with open(l) as data_file:
            data = json.load(data_file)

        detection_list = []

        for b in data['frames'][0]['objects']:
            if 'box2d' in b:
                x1 = b['box2d']['x1']
                y1 = b['box2d']['y1']
                x2 = b['box2d']['x2']
                y2 = b['box2d']['y2']

                obj_dict= { 'bbox': [x1, y1, x2, y2],
                            'name': image_id,
                            'category': b['category']
                          }

                detection_list.append(obj_dict)

        instances[image_id] = { 'path' : image_path,
                                'detections' : detection_list
                              }

    get_embeddings(instances)

    instances_flat = []
    for image_id in instances.keys():
        for d in instances[image_id]['detections']:
            instances_flat.append(d)

    np.save('bdd_gt_' + split + '_embeddings.npy', instances_flat)

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def bdd_prediction_embeddings(pred_file_path, image_dir, out_name):
    instances = {}

    with open(pred_file_path) as pred_file:
        pred_all = json.load(pred_file)

    detections = group_by_key(pred_all, 'name')

    for image_id in detections.keys():
        image_path = os.path.join(image_dir, image_id + '.jpg')
        instances[image_id] = { 'path': image_path,
                                 'detections': detections[image_id]
                               }
    get_embeddings(instances)

    instances_flat = []
    for image_id in instances.keys():
        for d in instances[image_id]['detections']:
            instances_flat.append(d)

    np.save(out_name, instances_flat)


if __name__ ==  "__main__":
    #bdd_gt_embeddings('val')
    #bdd_gt_embeddings('train')

    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    bdd_prediction_embeddings('pretrained.json', image_dir,
                              'bdd_val_pretrained_embeddings.npy')
    bdd_prediction_embeddings('supervised.json', image_dir,
                              'bdd_val_supervised_embeddings.npy')
