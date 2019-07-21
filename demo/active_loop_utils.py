from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import tensorflow as tf
import numpy as np
import cv2
import random
import math
import multiprocessing as mp
import subprocess
import argparse
import csv

from collections import defaultdict

from tensorflow.contrib import slim
sys.path.append(os.path.realpath('/n/pana/scratch/ravi/models/research/slim/'))

from nets import resnet_v2
from preprocessing import inception_preprocessing

from tqdm import tqdm
from shutil import copy, copyfile
from active_loop_components import active_loop_iteration

def get_class_name_to_id(dataset_dir):
    class_name_file = os.path.join(dataset_dir, 'labels',
                                   'class-descriptions-boxable.csv')

    class_name_to_id = {}
    with open(class_name_file) as class_names:
        reader = csv.reader(class_names, delimiter=',')
        for row in reader:
            class_name_to_id[row[1]] = row[0]
    return class_name_to_id

def preprocess_open_images_dataset(dataset_dir,
                                   list_images_by_class,
                                   list_instances_by_class):
    sub_dirs = [ 'train_' + i for i in \
                  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

    image_paths = []
    for sd in sub_dirs:
        image_paths = image_paths + glob.glob(os.path.join(dataset_dir, sd, '*.jpg'))

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    images_by_class = {}
    for cls_id in list_images_by_class.keys():
        images_by_class[cls_id] = set(list_images_by_class[cls_id])

    instances_by_img = {}
    for cls_id in list_instances_by_class.keys():
        for ins in list_instances_by_class[cls_id]:
            if ins['image_id'] not in instances_by_img:
                instances_by_img[ins['image_id']] = [ins]
            else:
                instances_by_img[ins['image_id']].append(ins)

    return image_id_to_path, images_by_class, instances_by_img

def get_search_imageset(search_image_dir):
    sample_paths = glob.glob(os.path.join(sample_dir, '*.jpg'))

    sample_id_to_path = {}
    for path in sample_paths[:1000]:
        sample_id_to_path[path.split('/')[-1].split('.')[0]] = path

    return sample_id_to_path

def get_num_images_by_class(images_by_class, image_ids):
    cls_counts = {}
    image_set = set(image_ids)
    for cls_id in images_by_class.keys():
        cls_counts[cls_id] = 0
        for im_id in images_by_class[cls_id]:
            if im_id in image_set:
                cls_counts[cls_id] = cls_counts[cls_id] + 1
    return cls_counts

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
        count = count + 1
        similarity_vis_per_img[image_id] = (similarity, count)
        if count > max_images:
            break
    return similarity_vis_per_img

def pick_query_instances(instances, image_id_to_path,
                         num_query_examples=5):
    curr_query_examples = 0
    sourced_images = []
    query_instances = []

    for ins in instances:
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

        min_size = 128 * 128

        if ((y2-y1) * (x2-x1) > min_size):
            patch = img[y1:y2, x1:x2, :]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

            query_instances.append((ins, patch, img))
            curr_query_examples = curr_query_examples + 1
            if curr_query_examples >= num_query_examples:
                break

    return query_instances

def get_embeddings(instances, model_name, return_dict):
    image_size = 299
    query_embeddings = []
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
            for ins, patch, vis_img in instances:
                scaled_img, logit_vals, embedding = sess.run([processed_image, logits, pool5], feed_dict={image: patch})
                query_embeddings.append((ins, patch, vis_img, embedding[0, 0, 0, :]))
    return_dict['query_embeddings'] = query_embeddings

def compute_instance_embeddings(query_instances, model_name):
    manager = mp.Manager()
    return_dict = manager.dict()
    proc = mp.Process(target=get_embeddings,
                          args=(query_instances, model_name,
                                return_dict))
    proc.start()
    proc.join()

    query_embeddings = return_dict['query_embeddings']
    return query_embeddings

def rank_by_similarity_to_query(query, image_embeddings):
    image_similarity = {}
    query = query / (np.linalg.norm(query, ord=2) + np.finfo(float).eps)
    for im in image_embeddings.keys():
        embeddings = image_embeddings[im]['embeddings']
        similarity = np.tensordot(embeddings, query, axes=1)
        image_similarity[im] = np.sum(similarity)

    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
    return sorted_images

def get_query_similarity(gpu_id, query_embedding, image_id_queue,
                         model_name, embeddings_queue, num_cutoff=50):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    norm_query_embeddings = {}

    query_embedding_size = query_embedding.shape[0]
    num_queries = 1
    query_mat = np.zeros((num_queries, query_embedding_size), dtype=np.float32)
    norm_embedding = query_embedding/ (np.linalg.norm(query_embedding, ord=2) + np.finfo(float).eps)
    query_mat[0] = norm_embedding

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

                query_embedding = query_mat[0]
                similarity_peaks = np.unravel_index(sorted_idxs[0],
                                                    similarity[0].shape)

                query_similar_embeddings = \
                            { 'embeddings': embedding[similarity_peaks].copy(),
                              'similarity': similarity[0].copy() }


                embeddings_queue.put((image_id, query_similar_embeddings))


def get_similarity(query_embedding, sample_id_to_path, model_name):

    image_id_queue = mp.Queue()
    embeddings_queue = mp.Queue()

    num_gpus = str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
    processes = []

    for gpu_id in range(num_gpus):
        proc = mp.Process(target=get_query_similarity,
                          args=(gpu_id, query_embedding,
                                image_id_queue, model_name,
                                embeddings_queue))
        proc.start()
        processes.append(proc)

    for im_id in sample_id_to_path.keys():
        image_id_queue.put((im_id, sample_id_to_path[im_id]))

    similar_embeddings = {}

    for _ in tqdm(sample_id_to_path.keys()):
        im, embeddings = embeddings_queue.get()
        similar_embeddings[im] = embeddings

    # Tell processes to exit
    for gpu_id in range(num_gpus):
        image_id_queue.put((None, None))
    for i in range(num_gpus):
        processes[i].join()

    return similar_embeddings

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

if __name__ == "__main__":
    np.random.seed(0)
    parser = argparse.ArgumentParser(description="Open Images Evaluation")
    parser.add_argument(
        "--category",
        default="Person",
        metavar="FILE",
        help="Class name",
    )

    parser.add_argument(
        "--experiment-dir",
        default=None,
        metavar="FILE",
        help="Directory for saving results",
    )

    parser.add_argument(
        "--command",
        default=None,
        metavar="FILE",
        help="Phase of active loop to run",
    )

    parser.add_argument(
        "--iteration",
        default=0,
        metavar="FILE",
        help="Phase of active loop to run",
    )

    args = parser.parse_args()
    assert(args.experiment_dir)
    assert(args.command)

    dataset_dir = '/n/pana/scratch/ravi/open_images'
    class_name_to_id = get_class_name_to_id(dataset_dir)

    assert(args.category in class_name_to_id)
    class_id = class_name_to_id[args.category]
    class_dir_name = args.category.replace(' ', '_')
    dest_dir = os.path.join(args.experiment_dir, class_dir_name)
    model_name = 'resnet_v2_50'

    if args.command == 'create_query':
        list_images_by_class = np.load('openimages_images_by_class.npy')[()]
        list_instances_by_class = np.load('openimages_instances_by_class.npy')[()]
        image_id_to_path, images_by_class, instances_by_img = \
                                preprocess_open_images_dataset(dataset_dir,
                                                               list_images_by_class,
                                                               list_instances_by_class)
        query_instances = pick_query_instances(list_instances_by_class[class_id],
                                               image_id_to_path)
        query_embeddings = compute_instance_embeddings(query_instances,
                                                       model_name)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
        query_vis_dir = os.path.join(dest_dir, 'query_images_vis')
        if not os.path.exists(query_vis_dir):
            os.makedirs(query_vis_dir)

        sum_embedding = None
        for ins, _, vis_img, embedding in query_embeddings:
            save_path = os.path.join(query_vis_dir, ins['image_id'] + '.png')
            cv2.imwrite(save_path, vis_img)
            if sum_embedding is None:
                sum_embedding = embedding
            else:
                sum_embedding = sum_embedding + embedding
        avg_embedding = sum_embedding/len(query_embeddings)

        state_save_path = os.path.join(dest_dir, 'iteration_0_state.npy')
        iteration_state = { 'positives': [],
                            'negatives': [],
                            'neutrals' : [],
                            'query_embedding': avg_embedding }

        np.save(state_save_path, iteration_state)
    elif args.command == 'run_iteration':
        iteration = int(args.iteration)
        iteration_prefix = 'iteration_' + str(iteration)
        iteration_state_file_name = iteration_prefix + '_state.npy'
        iteration_state_path = os.path.join(dest_dir, iteration_state_file_name)
        assert(os.path.exists(iteration_state_path))

        # Read iteration state
        iteration_state = np.load(iteration_state_path)[()]
        current_query = iteration_state['query_embedding']

        # Rank images by query
        sample_dir = '/n/pana/scratch/ravi/open_images_sub_sample_all_long_tail/'

        search_paths = get_search_imageset(sample_dir)
        query_similar_embeddings = get_similarity(current_query,
                                                  search_paths,
                                                  model_name)
        sorted_images = rank_by_similarity_to_query(current_query,
                                                    query_similar_embeddings)
        # visualize similar images
        similarity_vis = visualize_similarity(sorted_images,
                                              query_similar_embeddings,
                                              search_paths,
                                              max_images=100)
        # Prepare images for feedback
        num_feedback_images = 10
        window_size = 30
        feedback_requests = get_images_for_human_feedback(sorted_images,
                                                          num_feedback_images,
                                                          window_size)
        saved_embeddings = {}
        for im_id in feedback_requests:
            saved_embeddings[im_id] = query_similar_embeddings[im_id]['embeddings']

        for im_id in iteration_state['positives']:
            saved_embeddings[im_id] = query_similar_embeddings[im_id]['embeddings']

        for im_id in iteration_state['negatives']:
            saved_embeddings[im_id] = query_similar_embeddings[im_id]['embeddings']

        for im_id in iteration_state['neutrals']:
            saved_embeddings[im_id] = query_similar_embeddings[im_id]['embeddings']

        feedback_dir = os.path.join(dest_dir, iteration_prefix + '_feedback_images')
        if not os.path.exists(feedback_dir):
            os.makedirs(feedback_dir)

        for img_id in feedback_requests:
            copy(search_paths[img_id], feedback_dir)

        feedback_state = { 'ranked_images' : sorted_images,
                           'saved_embeddings' : saved_embeddings,
                           'similarity_vis' : similarity_vis,
                           'feedback_requests' : feedback_requests }

        feedback_state_path = os.path.join(dest_dir, iteration_prefix + '_feedback_state.npy')

        np.save(feedback_state_path, feedback_state)
    else:
        print('Unknown command')
