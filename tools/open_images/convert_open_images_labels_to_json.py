import os
import numpy as np
import json
import glob
import random
import csv
from collections import defaultdict
from tqdm import tqdm

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

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

    class_id_to_name = {}
    with open(class_name_file) as class_names:
        reader = csv.reader(class_names, delimiter=',')
        for row in reader:
            class_id_to_name[row[0]] = row[1]

    bbox_file = os.path.join(image_dir, 'labels', 'train-annotations-bbox.csv')

    instances_class_id = {}
    with open(bbox_file) as bboxes:
        reader = csv.reader(bboxes, delimiter=',')
        for row in reader:
            if row[0] == 'ImageID':
                continue
            if int(row[10]) > 0 or int(row[11]) > 0 or int(row[12]) > 0:
                continue
            instance = { 'category': class_id_to_name[row[2]],
                         'bbox': [ float(row[4]),
                                   float(row[5]),
                                   float(row[6]),
                                   float(row[7]) ],
                         'name': row[0]
                       }
            if row[2] not in instances_class_id:
                instances_class_id[row[2]] = [instance]
            else:
                instances_class_id[row[2]].append(instance)

    num_positives = 10000

    for class_id in instances_class_id.keys():
        ins_by_img = group_by_key(instances_class_id[class_id], 'name')

        image_ids = np.random.choice(list(ins_by_img.keys()),
                                     min(num_positives, len(ins_by_img.keys())), replace=False)
        classes = [ class_id_to_name[class_id] ]

        positive_instances = []
        for img_id in image_ids:
            positive_instances = positive_instances + ins_by_img[img_id]

        open_images_annotations = { 'image_ids' : image_ids.tolist(),
                                    'classes': classes,
                                    'annotations': positive_instances }

        cls_name = classes[0].replace(' ', '_')
        out_name = 'open_images_' + cls_name + '_' + str(min(num_positives, len(ins_by_img.keys()))) + '.json'
        with open(out_name, 'w') as fp:
            json.dump(open_images_annotations, fp)
