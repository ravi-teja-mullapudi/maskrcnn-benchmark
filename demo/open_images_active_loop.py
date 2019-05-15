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
import gc

from tqdm import tqdm
from collections import defaultdict

from tensorflow.contrib import slim
sys.path.append(os.path.realpath('/n/pana/scratch/ravi/models/research/slim/'))

from nets import resnet_v2
from preprocessing import inception_preprocessing
from bdd_active_loop import visualize_similarity

def get_query_similarity(query_embeddings, images, num_cutoff=50):
    norm_query_embeddings = {}
    for cls_id in query_embeddings.keys():
        norm_embedding = query_embeddings[cls_id] / (np.linalg.norm(query_embeddings[cls_id], ord=2) + np.finfo(float).eps)
        norm_query_embeddings[cls_id] = norm_embedding

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

        query_similar_embeddings = {}
        for cls_id in norm_query_embeddings.keys():
            query_similar_embeddings[cls_id] = {}

        with tf.Session() as sess:
            init_fn(sess)
            for im in tqdm(images):
                img = cv2.imread(im)
                input_img, embedding = sess.run([processed_image, postnorm], feed_dict={image: img})
                embedding = embedding / (np.expand_dims(np.linalg.norm(embedding, axis=3, ord=2), axis=3) + np.finfo(float).eps)
                for cls_id in norm_query_embeddings.keys():
                    query_embedding = norm_query_embeddings[cls_id]
                    similarity = np.tensordot(embedding, query_embedding, axes=1)
                    similarity_peaks = np.unravel_index(np.argsort(-similarity, axis=None),
                                                    similarity.shape)

                    similarity_sorted = similarity[similarity_peaks]
                    similarity_coords = [ np.expand_dims(c, axis=0) for c in \
                                      similarity_peaks ]
                    similarity_coords = np.transpose(np.concatenate(similarity_coords,
                                                                axis=0))

                    query_similar_embeddings[cls_id][im] = \
                            { 'locs': similarity_coords[:num_cutoff],
                            'embeddings': embedding[similarity_peaks][:num_cutoff].copy(),
                            'similarity': similarity }

        return query_similar_embeddings

def get_embeddings(instances):
    image_size = 299
    query_embeddings = {}
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

            for cls_id in instances.keys():
                ins, patch = instances[cls_id]
                scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], feed_dict={image: patch})
                query_embeddings[cls_id] = embedding[0, 0, 0, :]

    return query_embeddings

if __name__ == "__main__":
    np.random.seed(0)

    image_dir = '/n/pana/scratch/ravi/open_images'
    sub_dirs = [ 'train_' + i for i in \
                  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

    image_paths = []
    for sd in sub_dirs:
        image_paths = image_paths + glob.glob(os.path.join(image_dir, sd, '*.jpg'))

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    classes_of_interest = [ ('/m/01bjv', 'Bus'),
                            ('/m/02pv19', 'Stop sign'),
                            ('/m/012n7d', 'Ambulance'),
                            ('/m/01pns0', 'Fire hydrant'),
                            ('/m/0h2r6', 'Van'),
                            ('/m/07r04', 'Truck'),
                            ('/m/07jdr', 'Train'),
                            ('/m/015qbp', 'Parking meter'),
                            ('/m/01lcw4', 'Limousine'),
                            ('/m/0pg52', 'Taxi')
                          ]

    sample_dir = '/n/pana/scratch/ravi/open_images_sub_sample'
    sample_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))

    sample_id_to_path = {}
    for path in sample_paths:
        sample_id_to_path[path.split('/')[-1].split('.')[0]] = path

    images_by_class = np.load('openimages_images_by_class.npy')[()]
    instances_by_class = np.load('openimages_instances_by_class.npy')[()]

    query_vis_dir = '/n/pana/scratch/ravi/maskrcnn-benchmark/demo/open_images_queries'
    query_instances = {}
    # pick an instance from each of the classes of interest
    for cls_id, name in classes_of_interest:
        for ins in instances_by_class[cls_id]:
            img = cv2.imread(image_id_to_path[ins['image_id']])
            h = img.shape[0]
            w = img.shape[1]
            x1 = int(float(ins['bbox'][0]) * w)
            x2 = int(float(ins['bbox'][1]) * w)
            y1 = int(float(ins['bbox'][2]) * h)
            y2 = int(float(ins['bbox'][3]) * h)
            if ((y2-y1) * (x2-x1) > 128 * 128):
                patch = img[y1:y2, x1:x2, :]
                cls_name = name.replace(' ', '_')
                patch_path = os.path.join(query_vis_dir, cls_name + '_patch.png')
                cv2.imwrite(patch_path, patch)
                patch_source_path = os.path.join(query_vis_dir, cls_name + '_patch_source.png')
                cv2.imwrite(patch_source_path, img)
                query_instances[cls_id] = (ins, patch)
                break

    query_embeddings = get_embeddings(query_instances)
    query_similar_embeddings = get_query_similarity(query_embeddings, sample_paths[:500])

    output_dir = './open_images_similarity_vis_iter0'
    for cls_id, name in classes_of_interest:
        cls_output_dir = os.path.join(output_dir, name.replace(' ', '_'))
        if not os.path.exists(cls_output_dir):
            os.makedirs(cls_output_dir)
        visualize_similarity(query_embeddings[cls_id], query_similar_embeddings[cls_id],
                             cls_output_dir, max_images=100)
