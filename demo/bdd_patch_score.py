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

from tqdm import tqdm
from collections import defaultdict

from tensorflow.contrib import slim
sys.path.append(os.path.realpath('/n/pana/scratch/ravi/models/research/slim/'))

from nets import resnet_v2
from datasets import imagenet

from bdd_instance_clustering import filter_instances, group_by_key

def get_patch_score(query_embedding, images, output_dir):
    query_embedding = query_embedding / (np.linalg.norm(query_embedding, ord=2) + np.finfo(float).eps)
    with tf.Graph().as_default():
        image = tf.placeholder(tf.uint8, (None, None, 3))
        if image.dtype != tf.float32:
            processed_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        else:
            processed_image = image

        processed_image = tf.subtract(processed_image, 0.5)
        processed_image = tf.multiply(processed_image, 2.0)

        processed_image = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            postnorm, _ = resnet_v2.resnet_v2_101(processed_image, None,
                                                  is_training=False,
                                                  global_pool=False,
                                                  output_stride=8)
        init_fn = slim.assign_from_checkpoint_fn('resnet_v2_101.ckpt',
                                slim.get_model_variables('resnet_v2'))

        with tf.Session() as sess:
            init_fn(sess)
            similarity_min = float("inf")
            similarity_max = float("-inf")

            image_similarity = {}

            for im in tqdm(images):
                img = cv2.imread(im)
                input_img, embedding = sess.run([processed_image, postnorm], feed_dict={image: img})
                embedding = embedding / (np.expand_dims(np.linalg.norm(embedding, axis=3, ord=2), axis=3) + np.finfo(float).eps)
                similarity = np.tensordot(embedding, query_embedding, axes=1)
                similarity_min = min(np.min(similarity), similarity_min)
                similarity_max = max(np.max(similarity), similarity_max)
                #total_similarity = np.sum(similarity > 0.25)
                similarity_sorted = np.sort(-similarity, axis=None)
                total_similarity = np.sum(similarity_sorted[:50])
                image_similarity[im] = total_similarity

            sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
            print(similarity_min, similarity_max)
            print(sorted_images)
            count = 0
            for im, dist in tqdm(sorted_images):
                img = cv2.imread(im)
                input_img, embedding = sess.run([processed_image, postnorm], feed_dict={image: img})
                embedding = embedding / (np.expand_dims(np.linalg.norm(embedding, axis=3, ord=2), axis=3) + np.finfo(float).eps)
                similarity = np.tensordot(embedding, query_embedding, axes=1)
                #similarity = ((similarity - similarity_min)/(similarity_max - similarity_min)) * 255
                similarity = ((similarity > 0.25) * 255).astype(np.uint8)
                similarity = np.stack((similarity,)*3, axis=-1)
                similarity = cv2.resize(similarity[0], (img.shape[1], img.shape[0])).astype(np.uint8)
                similarity_vis = np.concatenate((img, similarity), axis=1)
                if output_dir is not None:
                    output_file_path = os.path.join(output_dir, 'similarity_vis_' + str(count) + '.png')
                    cv2.imwrite(output_file_path, similarity_vis)
                count = count + 1

if __name__ ==  "__main__":
    np.random.seed(0)

    instances_val = np.load('bdd_gt_val_embeddings.npy')[()]
    large_val_instances = filter_instances(instances_val,
                                           instances_per_class=100,
                                           min_size=64)

    output_dir = './similarity_vis_max'
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'

    large_val_instances = group_by_key(large_val_instances, 'category')
    query_instance = large_val_instances['traffic sign'][5]
    query_image_path = os.path.join(image_dir, query_instance['name'] + '.jpg')
    query_img = cv2.imread(query_image_path)

    x1 = int(query_instance['bbox'][0])
    x2 = int(query_instance['bbox'][2])
    y1 = int(query_instance['bbox'][1])
    y2 = int(query_instance['bbox'][3])

    query_patch = query_img[y1:y2, x1:x2, :]
    cv2.imwrite(os.path.join(output_dir, 'query.png'), query_patch)

    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))[0:1000]
    get_patch_score(query_instance['pool5_resnet_v2_101'], image_list, output_dir)
