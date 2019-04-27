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
    #instances = np.load('bdd_val_pretrained_10000_nn_categories.npy')[()]
    instances = np.load('bdd_val_gt_10000_nn_categories.npy')[()]
    filtered_proposals = []
    confident_proposals = []
    for ins in instances:
        if 'nn_train_categories' in ins:
            large_instances = large_instances + 1
            c = Counter(ins['nn_train_categories'])
            common_cls, count = c.most_common()[0]
            #if count > 3:
            #    discrepancies = discrepancies + 1
            #    print(ins['category'], ins['nn_train_categories'])
            #    ins['category'] = common_cls
            ins['category'] = common_cls
        ins['score'] = 1.0

    np.save('bdd_val_gt_reassigned_common_nn_categories.npy', instances)
