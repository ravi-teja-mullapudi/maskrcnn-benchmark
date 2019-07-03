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
            if int(row[10]) > 0 or int(row[11]) > 0 or int(row[12]) > 0:
                continue
            instance = { 'category': row[2],
                         'bbox': [row[4], row[5], row[6], row[7]],
                         'image_id': row[0]
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
                  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

    image_paths = []
    for sd in sub_dirs:
        image_paths = image_paths + glob.glob(os.path.join(image_dir, sd, '*.jpg'))

    image_id_to_path = {}
    for path in image_paths:
        image_id_to_path[path.split('/')[-1].split('.')[0]] = path

    class_name_file = os.path.join(image_dir, 'labels', 'class-descriptions-boxable.csv')
    bbox_file = os.path.join(image_dir, 'labels', 'validation-annotations-bbox.csv')

    instances_class_id, class_id_to_name = get_instances_per_class(class_name_file, bbox_file)
    images_class_id = {}

    for cls in instances_class_id.keys():
        for ins in instances_class_id[cls]:
            if cls in images_class_id:
                images_class_id[cls].add(ins['image_id'])
            else:
                images_class_id[cls] = set([ins['image_id']])

    rnd_paths = np.random.choice(image_paths, 49000, replace=False)
    rnd_image_ids = set()
    for path in rnd_paths:
        rnd_image_ids.add(path.split('/')[-1].split('.')[0])

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

    for cls_id, name in classes_of_interest:
        rnd_cls_ids = np.random.choice(list(images_class_id[cls_id]), 100, replace=False)
        rnd_image_ids.update(rnd_cls_ids)

    images_by_class = {}
    for cls_id in images_class_id.keys():
        images_by_class[cls_id] = list(images_class_id[cls_id])

    np.save('openimages_images_by_class.npy', images_by_class)
    np.save('openimages_instances_by_class.npy', instances_class_id)

    dest_dir = '/n/pana/scratch/ravi/open_images_sub_sample'
    for img_id in rnd_image_ids:
        copy(image_id_to_path[img_id], '/n/pana/scratch/ravi/open_images_sub_sample')
