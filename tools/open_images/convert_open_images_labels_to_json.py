import os
import numpy as np
import json
import glob
import random
import csv
from tqdm import tqdm

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
            instance = { 'category': class_id_to_name[row[2]],
                         'bbox': [ float(row[4]),
                                   float(row[5]),
                                   float(row[6]),
                                   float(row[7]) ],
                         'name': row[0]
                       }
            print(instance)
            exit(0)
            if row[2] not in instances_class_id:
                instances_class_id[row[2]] = [instance]
            else:
                instances_class_id[row[2]].append(instance)

    num_positives = 1000
    for class_id in instances_class_id.keys():
        positive_annotations = np.random.choice(instances_class_id[class_id],
                                                num_positives, replace=False)
        image_ids = []
        for ann in positive_annotations:
            if ann['name'] not in image_ids:
                image_ids.append(ann['name'])
        classes = [ class_id_to_name[class_id] ]
        open_images_annotations = { 'image_ids' : image_ids,
                                    'classes': classes,
                                    'annotations': positive_annotations.tolist() }

        cls_name = classes[0].replace(' ', '_')
        out_name = 'open_images_' + cls_name + '_' + str(num_positives) + '.json'
        with open(out_name, 'w') as fp:
            json.dump(open_images_annotations, fp)
        exit(0)
