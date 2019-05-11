from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import cv2

import selectivesearch
from bdd_visualize import draw_boxes

if __name__ == "__main__":
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    for im_path in image_list:
        img = cv2.imread(im_path)
        img_lbl, regions = selectivesearch.selective_search(img, scale=1024, sigma=0.9)
        proposals = []
        for r in regions[:500]:
            ins = {}
            x1 = r['rect'][0]
            y1 = r['rect'][1]
            x2 = r['rect'][2]
            y2 = r['rect'][3]
            ins['bbox'] = [x1, y1, x2, y2]
            ins['category'] = 'proposal'
            proposals.append(ins)
        vis_proposals = draw_boxes(img, proposals)
        cv2.imwrite('selecive_vis.png', vis_proposals)
        exit(0)
