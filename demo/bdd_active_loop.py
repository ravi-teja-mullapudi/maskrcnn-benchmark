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
from datasets import imagenet

from bdd_instance_clustering import filter_instances, group_by_key

def get_patch_score(query_embedding, images, num_cutoff=50):
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

        image_similar_embeddings = {}

        with tf.Session() as sess:
            init_fn(sess)

            for im in tqdm(images):
                img = cv2.imread(im)
                input_img, embedding = sess.run([processed_image, postnorm], feed_dict={image: img})
                embedding = embedding / (np.expand_dims(np.linalg.norm(embedding, axis=3, ord=2), axis=3) + np.finfo(float).eps)
                similarity = np.tensordot(embedding, query_embedding, axes=1)
                similarity_peaks = np.unravel_index(np.argsort(-similarity, axis=None),
                                                    similarity.shape)

                similarity_sorted = similarity[similarity_peaks]
                similarity_coords = [ np.expand_dims(c, axis=0) for c in \
                                      similarity_peaks ]
                similarity_coords = np.transpose(np.concatenate(similarity_coords,
                                                                axis=0))

                image_similar_embeddings[im] = { 'locs': similarity_coords[:num_cutoff],
                        'embeddings': embedding[similarity_peaks][:num_cutoff].copy(),
                        'similarity': similarity }

        return image_similar_embeddings

def visualize_similarity(query, image_similar_embeddings, output_dir):

    image_similarity = {}
    query = query / (np.linalg.norm(query, ord=2) + np.finfo(float).eps)
    for im in image_similar_embeddings.keys():
        embeddings = image_similar_embeddings[im]['embeddings']
        similarity = np.tensordot(embeddings, query, axes=1)
        image_similarity[im] = np.sum(similarity)

    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])

    count = 0
    for im, dist in tqdm(sorted_images):
        img = cv2.imread(im)
        similarity = image_similar_embeddings[im]['similarity']
        similarity = (similarity - np.min(similarity))/(np.max(similarity) - np.min(similarity) + np.finfo(float).eps)
        similarity = np.stack((similarity * 255,) * 3, axis=-1).astype(np.uint8)
        similarity = cv2.resize(similarity[0], (img.shape[1], img.shape[0])).astype(np.uint8)
        similarity_vis = np.concatenate((img, similarity), axis=1)
        if output_dir is not None:
            output_file_path = os.path.join(output_dir, 'similarity_vis_%03d'%(count) + '.png')
            cv2.imwrite(output_file_path, similarity_vis)
        count = count + 1

def get_positive_negative_embeddings(query,
                                     image_similar_embeddings,
                                     instances,
                                     num_samples,
                                     cls):
    image_similarity = {}
    query = query / (np.linalg.norm(query, ord=2) + np.finfo(float).eps)
    for im in image_similar_embeddings.keys():
        embeddings = image_similar_embeddings[im]['embeddings']
        similarity = np.tensordot(embeddings, query, axes=1)
        image_similarity[im] = np.sum(similarity)

    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
    positive_embeddings = []
    negative_embeddings = []

    for im, dist in tqdm(sorted_images[:num_samples]):
        embeddings = image_similar_embeddings[im]['embeddings']
        image_id = im.split('/')[-1].split('.')[0]
        if image_id in large_val_instances_image:
            instances = large_val_instances_image[image_id]
            instances_cat = group_by_key(instances, 'category')
            if cls in instances_cat:
                positive_embeddings.append(np.mean(embeddings, axis=0))
            else:
                negative_embeddings.append(np.mean(embeddings, axis=0))
        else:
            negative_embeddings.append(np.mean(embeddings, axis=0))

    return positive_embeddings, negative_embeddings

def get_human_box_level_supervision(image_similarity,
                                    image_similar_embeddings,
                                    instances, cls):
    pass

def get_positive_negative_instances(image_similarity,
                                    image_similar_embeddings,
                                    instances, cls):
    sorted_images = sorted(image_similarity.items(), key=lambda x: -x[1])
    postive_instances = {}

    for im, dist in tqdm(sorted_images):
        img = cv2.imread(im)
        h = img.shape[0]
        w = img.shape[1]
        similarity = image_similar_embeddings[im]['similarity']
        embedding_h = similarity.shape[1]
        embedding_w = similarity.shape[2]
        locs = image_similar_embeddings[im]['locs']

        image_id = im.split('/')[-1].split('.')[0]
        if image_id in large_val_instances_image:
            instances = large_val_instances_image[image_id]

            for loc_idx in range(len(locs)):
                in_labeled_box = False
                scaled_h = float(h)/embedding_h * locs[loc_idx]
                scaled_w = float(w)/embedding_w * locs[loc_idx]
                for ins in instances:
                    x1 = ins['bbox'][0]
                    y1 = ins['bbox'][1]
                    x2 = ins['bbox'][2]
                    y2 = ins['bbox'][3]
                    if scaled_h > x1 and scaled_h < x2 and \
                        scaled_w > y1 and scaled_w < y2:
                        if ins['category'] == cls:
                            postive_instances.append(ins)
                        else:
                            negative_instances.append(ins)
                        in_labeled_box = True

                    print(loc)

# TODO: Implement a clustering based segmentation of the image and only store cluster
# centers and corresponding weights

if __name__ ==  "__main__":
    np.random.seed(0)

    instances_val = np.load('bdd_gt_val_embeddings.npy')[()]
    large_val_instances = filter_instances(instances_val,
                                           instances_per_class=100000,
                                           min_size=64)

    output_dir = './similarity_vis_max'
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'

    large_val_instances_cat = group_by_key(large_val_instances, 'category')
    large_val_instances_image = group_by_key(large_val_instances, 'name')

    query_instance = large_val_instances_cat['traffic sign'][5]
    query_image_path = os.path.join(image_dir, query_instance['name'] + '.jpg')
    query_img = cv2.imread(query_image_path)

    traffic_sign_images = 0
    for im in large_val_instances_image.keys():
        im_cat = group_by_key(large_val_instances_image[im], 'category')
        if 'traffic sign' in im_cat:
            traffic_sign_images = traffic_sign_images + 1

    x1 = int(query_instance['bbox'][0])
    x2 = int(query_instance['bbox'][2])
    y1 = int(query_instance['bbox'][1])
    y2 = int(query_instance['bbox'][3])

    query_patch = query_img[y1:y2, x1:x2, :]
    cv2.imwrite(os.path.join(output_dir, 'query.png'), query_patch)

    # Compute embedding locations which are similar to query in each image
    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))[0:1000]
    query = query_instance['pool5_resnet_v2_101']
    image_similar_embeddings = get_patch_score(query, image_list)

    visualize_similarity(query, image_similar_embeddings, output_dir)

    pos_embeddings, neg_embeddings = get_positive_negative_embeddings(query,
                                                        image_similar_embeddings,
                                                        large_val_instances_image,
                                                        25, 'traffic sign')
    print(len(pos_embeddings), len(neg_embeddings))

    refined_query = sum(pos_embeddings)/len(pos_embeddings) - sum(neg_embeddings)/len(neg_embeddings)
    image_similar_embeddings = get_patch_score(refined_query, image_list)
    output_dir = './similarity_vis_refined'
    visualize_similarity(refined_query, image_similar_embeddings, output_dir)

    exit(0)

    positve_samples = []
    for im, dist in sorted_images[:25]:
        image_id = im.split('/')[-1].split('.')[0]
        if image_id in large_val_instances_image:
            instances = large_val_instances_image[image_id]
            instances_cat = group_by_key(instances, 'category')
            if 'traffic sign' in instances_cat:
                positve_samples = positve_samples + instances_cat['traffic sign']

    random_samples = []
    for im in image_list[:25]:
        image_id = im.split('/')[-1].split('.')[0]
        if image_id in large_val_instances_image:
            instances = large_val_instances_image[image_id]
            instances_cat = group_by_key(instances, 'category')
            if 'traffic sign' in instances_cat:
                random_samples = random_samples + instances_cat['traffic sign']

    print(len(random_samples), len(positve_samples))
