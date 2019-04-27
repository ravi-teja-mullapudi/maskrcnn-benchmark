from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import numpy as np
import time
import cv2
import random
import json

from tqdm import tqdm
from collections import Counter, defaultdict
from bdd_analysis import group_by_key
from bdd_large_object_analysis import draw_boxes

if __name__ == "__main__":
    discrepancies = 0
    large_instances = 0
    instances = np.load('bdd_val_pretrained_10000_nn_categories.npy')[()]
    filtered_proposals = []
    confident_proposals = []
    for ins in instances:
        if ins['score'] >= 0.5:
            confident_proposals.append(ins)
        if 'nn_train_categories' in ins and ins['score'] < 0.5 and ins['score'] > 0.25:
            large_instances = large_instances + 1
            c = Counter(ins['nn_train_categories'])
            common_cls, count = c.most_common()[0]
            if count > 3:
                discrepancies = discrepancies + 1
                filtered_proposals.append(ins)
                ins['category'] = common_cls

    per_im_proposals = group_by_key(filtered_proposals, 'name')
    per_im_conf_proposals = group_by_key(confident_proposals, 'name')

    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    output_dir = '/n/pana/scratch/ravi/maskrcnn-benchmark/demo/proposal_vis'
    for image_id in tqdm(per_im_proposals.keys()):
        image_path = os.path.join(image_dir, image_id + '.jpg')
        img = cv2.imread(image_path)
        proposal_vis = draw_boxes(img, per_im_proposals[image_id])
        conf_proposal_vis = draw_boxes(img, per_im_conf_proposals[image_id])
        vis = np.concatenate((proposal_vis, conf_proposal_vis), axis=0)
        cv2.imwrite(os.path.join(output_dir, image_id + '.png'), vis)
    #np.save('bdd_val_pretrained_reassigned_nn_categories.npy', instances)
