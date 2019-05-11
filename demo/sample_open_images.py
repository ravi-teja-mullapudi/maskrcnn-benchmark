from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import numpy as np
import csv
from shutil import copy

def get_instances_per_class(class_name_file, bbox_file):
    class_id_to_name = {}
    with open(class_name_file) as class_names:
        reader = csv.reader(class_names, delimiter=',')
        for row in reader:
            class_id_to_name[row[0]] = row[1]

    instances_class_id = {}
    with open(bbox_file) as bboxes:
        reader = csv.reader(bboxes, delimiter=',')
        for row in reader:
            if row[0] == 'ImageID':
                continue
            instance = { 'category': row[2],
                         'bbox': [row[4], row[5], row[6], row[7]],
                         'image_id': row[1]
                       }
            if row[2] not in instances_class_id:
                instances_class_id[row[2]] = [instance]
            else:
                instances_class_id[row[2]].append(instance)

    return instances_class_id, class_id_to_name

if __name__ ==  "__main__":
    np.random.seed(0)
    image_dir = '/n/pana/scratch/ravi/open_images'
    sub_dirs = [ 'train_' + i for i in \
                  [ '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

    image_paths = []
    for sd in sub_dirs:
        image_paths = image_paths + glob.glob(os.path.join(image_dir, sd, '*.jpg'))

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    vis_dir = '/n/pana/scratch/ravi/maskrcnn-benchmark/demo/open_images_vis'
    rnd_paths = np.random.choice(image_paths, 1000, replace=False)
    for path in rnd_paths:
        copy(path, vis_dir)

    exit(0)

    class_name_file = os.path.join(image_dir, 'labels', 'class-descriptions-boxable.csv')
    bbox_file = os.path.join(image_dir, 'labels', 'train-annotations-bbox.csv')

    instances_class_id, class_id_to_name = get_instances_per_class(class_name_file, bbox_file)
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
