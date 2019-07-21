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
import math
import multiprocessing as mp
import subprocess
import argparse
import csv

from tqdm import tqdm
from collections import defaultdict

from tensorflow.contrib import slim
sys.path.append(os.path.realpath('/n/pana/scratch/ravi/models/research/slim/'))

from nets import resnet_v2
from preprocessing import inception_preprocessing

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def rank_by_similarity_to_query(query, image_embeddings):
    image_similarity = {}
    query = query / (np.linalg.norm(query, ord=2) + np.finfo(float).eps)
    for im in image_embeddings.keys():
        embeddings = image_embeddings[im]['embeddings']
        similarity = np.tensordot(embeddings, query, axes=1)
        image_similarity[im] = np.sum(similarity)

    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
    return sorted_images

def get_images_for_human_feedback(sorted_images,
                                  num_feedback_images,
                                  window_size):
    get_spaced_idxs = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    spaced_idxs = get_spaced_idxs(num_feedback_images, window_size)

    feedback_images = []
    for idx in spaced_idxs:
        image_id, _ = sorted_images[idx]
        feedback_images.append(image_id)
    return feedback_images

def visualize_similarity(sorted_images, image_embeddings,
                         sample_id_to_path, max_images=100):
    count = 0
    similarity_vis_per_img = {}
    for image_id, dist in tqdm(sorted_images[:min(max_images, len(sorted_images))]):
        img = cv2.imread(sample_id_to_path[image_id])
        similarity = image_embeddings[image_id]['similarity']
        similarity = (similarity - np.min(similarity))/(np.max(similarity) - np.min(similarity) + np.finfo(float).eps)
        similarity = np.stack((similarity * 255,) * 3, axis=-1).astype(np.uint8)
        similarity = cv2.resize(similarity[0], (img.shape[1], img.shape[0])).astype(np.uint8)
        similarity_vis = np.concatenate((img, similarity), axis=1)
        count = count + 1
        similarity_vis_per_img[image_id] = (similarity_vis, count)
        if count > max_images:
            break
    return similarity_vis

def get_query_similarity(gpu_id, query_embeddings, image_id_queue,
                         model_name, embeddings_queue, num_cutoff=50):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    norm_query_embeddings = {}

    query_embedding_size = query_embeddings.values()[0].shape[0]
    num_queries = len(query_embeddings.keys())
    query_mat = np.zeros((num_queries, query_embedding_size), dtype=np.float32)
    for count, cls_id in enumerate(query_embeddings.keys()):
        norm_embedding = query_embeddings[cls_id] / (np.linalg.norm(query_embeddings[cls_id], ord=2) + np.finfo(float).eps)
        norm_query_embeddings[cls_id] = count
        query_mat[count] = norm_embedding

    with tf.Graph().as_default():
        image = tf.placeholder(tf.uint8, (None, None, 3))
        query_in = tf.placeholder(tf.float32, (num_queries, query_embedding_size))
        if image.dtype != tf.float32:
            processed_image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        else:
            processed_image = image

        processed_image = tf.subtract(processed_image, 0.5)
        processed_image = tf.multiply(processed_image, 2.0)

        processed_image = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            if model_name == 'resnet_v2_101':
                postnorm, _ = resnet_v2.resnet_v2_101(processed_image, None,
                                                      is_training=False,
                                                      global_pool=False,
                                                      output_stride=8)
            elif model_name == 'resnet_v2_50':
                postnorm, _ = resnet_v2.resnet_v2_50(processed_image, None,
                                                      is_training=False,
                                                      global_pool=False,
                                                      output_stride=8)
            else:
                print('Unknown model')
                exit(0)
            postnorm = tf.nn.l2_normalize(postnorm, axis=3)
            query_similarity = tf.tensordot(query_in, postnorm, axes=[[1], [3]])
            query_sorted_idxs = tf.argsort(tf.reshape(query_similarity, [num_queries, -1]),
                                           direction='DESCENDING')
            query_sorted_idxs = query_sorted_idxs[:, :num_cutoff]

        if model_name == 'resnet_v2_101':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v2_101.ckpt',
                                    slim.get_model_variables('resnet_v2'))
        elif model_name == 'resnet_v2_50':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v2_50.ckpt',
                                    slim.get_model_variables('resnet_v2'))

        with tf.Session() as sess:
            init_fn(sess)
            while True:
                image_id, im_path  = image_id_queue.get()
                if image_id == None:
                    break
                img = cv2.imread(im_path)
                height, width = img.shape[:2]
                max_image_size = 1920 * 1080
                if height * width > max_image_size:
                    scale_factor = math.sqrt((height * width)/max_image_size)
                    img = cv2.resize(img, (int(width/scale_factor), int(height/scale_factor)))

                input_img, embedding, similarity, sorted_idxs = \
                        sess.run([processed_image, postnorm, query_similarity, query_sorted_idxs],
                                  feed_dict={image: img, query_in: query_mat})

                query_similar_embeddings = {}

                for cls_id in norm_query_embeddings.keys():
                    cls_pos = norm_query_embeddings[cls_id]
                    query_embedding = query_mat[cls_pos]
                    similarity_peaks = np.unravel_index(sorted_idxs[cls_pos],
                                                        similarity[cls_pos].shape)

                    query_similar_embeddings[cls_id] = \
                            { 'embeddings': embedding[similarity_peaks].copy(),
                              'similarity': similarity[cls_pos].copy() }


                embeddings_queue.put((image_id, query_similar_embeddings))

def get_similarity(query_embeddings, sample_id_to_path, model_name):

    image_id_queue = mp.Queue()
    embeddings_queue = mp.Queue()

    num_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    processes = []

    for gpu_id in range(num_gpus):
        proc = mp.Process(target=get_query_similarity,
                          args=(gpu_id, query_embeddings,
                                image_id_queue, model_name,
                                embeddings_queue))
        proc.start()
        processes.append(proc)

    for im_id in sample_id_to_path.keys():
        image_id_queue.put((im_id, sample_id_to_path[im_id]))

    cls_similar_embeddings = {}
    for cls_id in query_embeddings.keys():
        cls_similar_embeddings[cls_id] = {}

    for _ in tqdm(sample_id_to_path.keys()):
        im, similar_embeddings = embeddings_queue.get()
        for cls_id in similar_embeddings.keys():
            cls_similar_embeddings[cls_id][im] = similar_embeddings[cls_id]

    # Tell processes to exit
    for gpu_id in range(num_gpus):
        image_id_queue.put((None, None))
    for i in range(num_gpus):
        processes[i].join()

    return cls_similar_embeddings

def get_embeddings(instances, model_name, return_dict):
    image_size = 299
    query_embeddings = {}
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.Graph().as_default():
        image = tf.placeholder(tf.uint8, (None, None, 3))
        processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
        processed_image = tf.expand_dims(processed_image, 0)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            if model_name == 'resnet_v2_101':
                logits, _ = resnet_v2.resnet_v2_101(processed_image, 1001, is_training=False)
                pool5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_101/pool5:0")
            elif model_name == 'resnet_v2_50':
                logits, _ = resnet_v2.resnet_v2_50(processed_image, 1001, is_training=False)
                pool5 = tf.get_default_graph().get_tensor_by_name("resnet_v2_50/pool5:0")
            else:
                print("Unknown model")
                exit(0)
        if model_name == 'resnet_v2_101':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v2_101.ckpt',
                                slim.get_model_variables('resnet_v2'))
        elif model_name == 'resnet_v2_50':
            init_fn = slim.assign_from_checkpoint_fn('resnet_v2_50.ckpt',
                                slim.get_model_variables('resnet_v2'))

        with tf.Session() as sess:
            init_fn(sess)
            for ins_id in instances.keys():
                ins, patch = instances[ins_id]
                scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], feed_dict={image: patch})
                query_embeddings[ins_id] = embedding[0, 0, 0, :]
    return_dict['query_embeddings'] = query_embeddings

def compute_query_embedding(classes, query_embeddings,
        num_queries, user_positives, user_negatives,
        user_neutral):
    avg_query_embeddings = {}
    for cls_id in classes:
        sum_embedding = None
        for i in range(num_queries):
            embedding = query_embeddings[(cls_id, i)]
            if sum_embedding is None:
                sum_embedding = embedding
            else:
                sum_embedding = sum_embedding + embedding
        avg_embedding = sum_embedding/num_queries
        avg_query_embeddings[cls_id] = avg_embedding
    return avg_query_embeddings

def active_loop_iteration(query_instances, num_queries,
                          user_positives, user_negatives,
                          user_neutral, model_name,
                          image_paths):
    manager = mp.Manager()
    return_dict = manager.dict()
    proc = mp.Process(target=get_embeddings,
                          args=(query_instances, model_name,
                                return_dict))
    proc.start()
    proc.join()

    query_embeddings = return_dict['query_embeddings']

    classes = set()
    for cls_id, _ in query_instances.keys():
        if cls_id not in classes:
            classes.add(cls_id)

    # Compute query embedding using user positives, negatives amd neutrals
    current_query = \
           compute_query_embedding(classes,
                                   query_embeddings,
                                   num_queries,
                                   user_positives,
                                   user_negatives,
                                   user_neutral)
    query_similar_embeddings = \
            get_similarity(current_query, image_paths,
                           model_name)
    ranked_images = {}
    feedback_images = {}
    similarity_vis_images = {}

    for cls_id in classes:
        # Rank images
        sorted_images = \
            rank_by_similarity_to_query(current_query[cls_id],
                                        query_similar_embeddings[cls_id])

        ranked_images[cls_id] = sorted_images

        # visualize similar images
        similarity_vis = visualize_similarity(sorted_images,
                             query_similar_embeddings[cls_id],
                             image_paths,
                             max_images=100)
        num_feedback_images = 10
        window_size = 30
        feedback_requests = get_images_for_human_feedback(sorted_images,
                                                          num_feedback_images,
                                                          window_size)
        feeback_embeddings = {}
        for im_id in feedback_requests:
            feeback_embeddings[im_id] = \
                    query_similar_embeddings[cls_id][im_id]['embeddings']
        feedback_images[cls_id] = feeback_embeddings
        similarity_vis_images[cls_id] = similarity_vis

    return ranked_images, feedback_images, similarity_vis_images
    #np.save(os.path.join(output_dir, 'ranked_images.npy'), ranked_images)
    #np.save(os.path.join(output_dir, 'query_instances.npy'), query_instances)
    #np.save(os.path.join(output_dir, 'user_positives.npy'), user_positives)
    #np.save(os.path.join(output_dir, 'user_negatives.npy'), user_negatives)
    #np.save(os.path.join(output_dir, 'user_neutral.npy'), user_neutral)
