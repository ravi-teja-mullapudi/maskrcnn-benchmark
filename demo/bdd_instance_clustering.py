from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
sys.setrecursionlimit(2000)

import numpy as np
import time
import cv2
import random
import json

from tqdm import tqdm
from scipy import spatial
from sklearn.preprocessing import normalize
from collections import defaultdict, Counter

def build_kd_tree(instances):
    embedding_size = instances[0]['pool5_resnet_v2_101'].shape[0]
    num_instances = len(instances)
    embeddings = np.zeros((num_instances, embedding_size), dtype=np.float32)
    for idx in range(len(instances)):
        embeddings[idx] = instances[idx]['pool5_resnet_v2_101']
    #embeddings = normalize(embeddings, axis=1)
    kdtree = spatial.KDTree(embeddings)
    return kdtree

def get_patch(image_path, bbox, patch_shape):
    img = cv2.imread(image_path)
    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])
    patch = cv2.resize(img[y1:y2, x1:x2, :], patch_shape)
    return patch

def visualize_instances(instances, image_dir):
    crops = []
    crop_size = 299
    for ins in instances:
        image_id = ins['name']
        image_path = os.path.join(image_dir, image_id + '.jpg')
        crop = get_patch(image_path, ins['bbox'], (crop_size, crop_size))
        crops.append(crop)
    vis_img = np.concatenate(crops, axis=1)
    return vis_img

def test():
    instances = np.load('bdd_gt_val_embeddings.npy')[()]
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    output_dir = './nearest_neighbors'
    large_train_instances = filtered_instances(instances)
    for ins in range(100):
        q = normalize(np.expand_dims(large_instances[ins]['pool5_resnet_v2_101'], 0))[0]
        distances, idxs = kdtree.query([q], k=5)
        nearest_instances = [ large_instances[i] for i in idxs[0] ]
        vis_instances = [ large_instances[ins] ] + nearest_instances
        vis_image = visualize_instances(vis_instances, image_dir)
        save_path = os.path.join(output_dir, str(ins) + '.png')
        cv2.imwrite(save_path, vis_image)

def filter_instances(instances, min_size=64, instances_per_class=10000):
    filtered_instances = []
    for ins in instances:
        x1 = ins['bbox'][0]
        y1 = ins['bbox'][1]
        x2 = ins['bbox'][2]
        y2 = ins['bbox'][3]
        if (x2-x1) * (y2-y1) > min_size * min_size:
            filtered_instances.append(ins)
    cat_instances = group_by_key(filtered_instances, 'category')

    balanced_instances = []
    if instances_per_class is not None:
        for cls in cat_instances.keys():
            num_instances_cls = len(cat_instances[cls])
            idxs = np.random.choice(num_instances_cls, min(instances_per_class, num_instances_cls), replace=False)
            balanced_instances = balanced_instances + [ cat_instances[cls][idx] for idx in idxs ]
            print(cls, len(cat_instances[cls]), len(balanced_instances))
    else:
        balanced_instances = filtered_instances

    return balanced_instances

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

def nearest_neighbor_labels():
    np.random.seed(0)

    instances = np.load('bdd_gt_train_embeddings.npy')[()]
    image_train_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/train'
    image_val_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val'
    output_dir = './nearest_neighbors_cosine'

    large_train_instances = filter_instances(instances)
    #kdtree = build_kd_tree(large_train_instances)
    embedding_size = large_train_instances[0]['pool5_resnet_v2_101'].shape[0]

    num_instances = len(large_train_instances)
    embeddings = np.zeros((num_instances, embedding_size), dtype=np.float32)
    for idx in range(len(large_train_instances)):
        embeddings[idx] = large_train_instances[idx]['pool5_resnet_v2_101']

    instances_val = np.load('bdd_gt_val_embeddings.npy')[()]
    large_val_instances = filter_instances(instances_val, instances_per_class=None)

    large_class_val_instances = group_by_key(large_val_instances, 'category')

    for cls in large_class_val_instances.keys():
        for val_ins in range(min(25, len(large_class_val_instances[cls]))):
            q = large_class_val_instances[cls][val_ins]['pool5_resnet_v2_101']
            #distances, idxs = kdtree.query([q], k=5)
            #nearest_train_instances = [ large_train_instances[i] for i in idxs[0] ]
            scores = embeddings.dot(q)
            idxs = (-scores).argsort()[:5]
            nearest_train_instances = [ large_train_instances[i] for i in idxs ]
            neighbors_image = visualize_instances(nearest_train_instances, image_train_dir)
            query_image = visualize_instances([large_class_val_instances[cls][val_ins]], image_val_dir)
            vis_image = np.concatenate((query_image, neighbors_image), axis=1)
            save_path = os.path.join(output_dir, cls + '_' + str(val_ins) + '.png')
            cv2.imwrite(save_path, vis_image)

def nearest_neighbor_instance_labeling():
    np.random.seed(0)
    output_dir = './nearest_neighbors_discrepancies'
    image_train_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/train'
    image_val_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val'

    instances = np.load('bdd_gt_train_embeddings.npy')[()]
    large_train_instances = filter_instances(instances)
    #kdtree = build_kd_tree(large_train_instances)

    embedding_size = large_train_instances[0]['pool5_resnet_v2_101'].shape[0]

    num_instances = len(large_train_instances)
    embeddings = np.zeros((num_instances, embedding_size), dtype=np.float32)
    for idx in range(len(large_train_instances)):
        embeddings[idx] = large_train_instances[idx]['pool5_resnet_v2_101']

    #instances_val = np.load('bdd_val_pretrained_embeddings.npy')[()]
    instances_val = np.load('bdd_gt_val_embeddings.npy')[()]

    for ins in tqdm(range(len(instances_val))):
        if 'pool5_resnet_v2_101' in instances_val[ins]:
            q = instances_val[ins]['pool5_resnet_v2_101']
            scores = embeddings.dot(q)
            idxs = (-scores).argsort()[:5]
            #distances, idxs = kdtree.query([q], k=5)
            nearest_train_categories = [ large_train_instances[i]['category'] for i in idxs ]
            instances_val[ins]['nn_train_categories'] = nearest_train_categories
            instances_val[ins]['nn_train_distances'] = scores[idxs]
            #c = Counter(nearest_train_categories)
            #common_cls, count = c.most_common()[0]
            #if count > 3 and common_cls != instances_val[ins]['category'] and instances_val[ins]['score'] > 0.5:
            #    nearest_train_instances = [ large_train_instances[i] for i in idxs ]
            #    neighbors_image = visualize_instances(nearest_train_instances, image_train_dir)
            #    query_image = visualize_instances([instances_val[ins]], image_val_dir)
            #    vis_image = np.concatenate((query_image, neighbors_image), axis=1)
            #    save_path = os.path.join(output_dir,
            #           instances_val[ins]['category'] + '_' + common_cls + '_' + str(ins) + '.png')
            #    cv2.imwrite(save_path, vis_image)


    np.save('bdd_val_gt_10000_nn_categories.npy', instances_val)

if __name__ == "__main__":
    #nearest_neighbor_labels()
    nearest_neighbor_instance_labeling()
