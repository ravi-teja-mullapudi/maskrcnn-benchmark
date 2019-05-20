from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import tensorflow as tf
import numpy as np
import time
import cv2
import random
import math
import multiprocessing as mp
import subprocess

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

def rank_by_query(query, image_similar_embeddings):
    image_similarity = {}
    query = query / (np.linalg.norm(query, ord=2) + np.finfo(float).eps)
    for im in image_similar_embeddings.keys():
        embeddings = image_similar_embeddings[im]['embeddings']
        similarity = np.tensordot(embeddings, query, axes=1)
        image_similarity[im] = np.sum(similarity)

    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
    return sorted_images

def get_postives_and_negatives(sorted_images, gt_instances, cls,
                               num_samples, window_size):

    get_spaced_idxs = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    spaced_idxs = get_spaced_idxs(num_samples, window_size)
    user_positive_samples = []
    user_negative_samples = []

    positive_idxs = []
    negative_idxs = []

    for idx in spaced_idxs:
        image_id, dist = sorted_images[idx]
        if image_id in gt_instances:
            instances = gt_instances[image_id]
            instances_cat = group_by_key(instances, 'category')
            if cls in instances_cat:
                user_positive_samples.append(image_id)
                positive_idxs.append(idx)
            else:
                user_negative_samples.append(image_id)
                negative_idxs.append(idx)

    prev_idx_positive = True
    max_streak_positive = -1
    for idx in spaced_idxs:
        if idx in positive_idxs and prev_idx_positive:
            max_streak_positive = idx
        else:
            prev_idx_positive = False

    inferred_positive_samples = []
    if max_streak_positive > 0:
        for idx in range(0, max_streak_positive):
            image_id, _ = sorted_images[idx]
            if image_id not in user_negative_samples and \
               image_id not in user_positive_samples:
                   inferred_positive_samples.append(image_id)

    prev_idx_negative = True
    min_streak_negative = -1
    for idx in spaced_idxs[::-1]:
        if idx in negative_idxs and prev_idx_negative:
            min_streak_negative = idx
        else:
            prev_idx_negative = False

    inferred_negative_samples = []
    if min_streak_negative > 0:
        for idx in range(window_size, min_streak_negative, -1):
            image_id, _ = sorted_images[idx]
            if image_id not in user_negative_samples and \
               image_id not in user_positive_samples:
                   inferred_negative_samples.append(image_id)

    # Remove user labels from inferred labels
    for image_id in user_negative_samples:
        if image_id in inferred_positive_samples:
            inferred_positive_samples.remove(image_id)
        if image_id in inferred_negative_samples:
            inferred_negative_samples.remove(image_id)

    for image_id in user_positive_samples:
        if image_id in inferred_positive_samples:
            inferred_positive_samples.remove(image_id)
        if image_id in inferred_negative_samples:
            inferred_negative_samples.remove(image_id)

    print('user positives', user_positive_samples)
    print('user negatives', user_negative_samples)
    print('inferred positives', inferred_positive_samples)
    print('inferred negatives', inferred_negative_samples)

def visualize_similarity(sorted_images, image_similar_embeddings,
                         sample_id_to_path, output_dir, max_images=100):
    count = 0
    for image_id, dist in tqdm(sorted_images[:min(max_images, len(sorted_images))]):
        img = cv2.imread(sample_id_to_path[image_id])
        similarity = image_similar_embeddings[image_id]['similarity']
        similarity = (similarity - np.min(similarity))/(np.max(similarity) - np.min(similarity) + np.finfo(float).eps)
        similarity = np.stack((similarity * 255,) * 3, axis=-1).astype(np.uint8)
        similarity = cv2.resize(similarity[0], (img.shape[1], img.shape[0])).astype(np.uint8)
        similarity_vis = np.concatenate((img, similarity), axis=1)
        if output_dir is not None:
            output_file_path = os.path.join(output_dir, 'similarity_vis_%03d'%(count) + '.png')
            cv2.imwrite(output_file_path, similarity_vis)
        count = count + 1
        if count > max_images:
            break

def get_recall(sorted_images, gt_instances, cls, max_queries=100, step=5):
    recalls = [ 0 for _ in range(0, max_queries, step) ]
    count = 0
    for im, dist in sorted_images[:max_queries]:
        image_id = im.split('/')[-1].split('.')[0]
        if image_id in gt_instances:
            instances = gt_instances[image_id]
            instances_cat = group_by_key(instances, 'category')
            if cls in instances_cat:
                recalls[int(count/step)] = recalls[int(count/step)] + 1
        count = count + 1
    return recalls

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
            for cls_id in instances.keys():
                ins, patch = instances[cls_id]
                scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], feed_dict={image: patch})
                query_embeddings[cls_id] = embedding[0, 0, 0, :]
    return_dict['query_embeddings'] = query_embeddings

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
    for path in sample_paths[:5000]:
        sample_id_to_path[path.split('/')[-1].split('.')[0]] = path

    images_by_class = np.load('openimages_images_by_class.npy')[()]
    instances_by_class = np.load('openimages_instances_by_class.npy')[()]

    instances_by_img = {}
    for cls_id in instances_by_class.keys():
        for ins in instances_by_class[cls_id]:
            if ins['image_id'] not in instances_by_img:
                instances_by_img[ins['image_id']] = [ins]
            else:
                instances_by_img[ins['image_id']].append(ins)

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

    model_name = 'resnet_v2_50'
    manager = mp.Manager()
    return_dict = manager.dict()
    proc = mp.Process(target=get_embeddings,
                          args=(query_instances, model_name,
                                return_dict))
    proc.start()
    proc.join()

    query_embeddings = return_dict['query_embeddings']
    num_active_iterations = 3
    for it in range(num_active_iterations):
        query_similar_embeddings = get_similarity(query_embeddings, sample_id_to_path,
                                                  model_name)

        output_dir = './open_images_similarity_vis_iter' + str(it)
        ranked_images = {}
        initial_window = 30
        num_samples = 10
        for cls_id, name in classes_of_interest:
            cls_output_dir = os.path.join(output_dir, name.replace(' ', '_'))
            if not os.path.exists(cls_output_dir):
                os.makedirs(cls_output_dir)

            # Rank images
            sorted_images = rank_by_query(query_embeddings[cls_id],
                                          query_similar_embeddings[cls_id])
            # visualize similar images
            visualize_similarity(sorted_images,
                                 query_similar_embeddings[cls_id],
                                 sample_id_to_path, cls_output_dir,
                                 max_images=100)
            # Compute recalls
            recalls = get_recall(sorted_images, instances_by_img, cls_id)
            print(name, recalls)

            # Get positives and negatives from user
            get_postives_and_negatives(sorted_images, instances_by_img, cls_id,
                                       num_samples, initial_window)
            # Update embeddings
        exit(0)
