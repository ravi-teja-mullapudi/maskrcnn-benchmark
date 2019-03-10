import os
import numpy as np
import json
import glob
from tqdm import tqdm

split = 'train'
bdd_labels_path = '/n/pana/scratch/ravi/bdd/bdd100k/labels/100k'
bdd_labels_list = glob.glob(os.path.join(bdd_labels_path, split, '*.json'))

image_ids = []
classes = []
annotations = []

for l in tqdm(bdd_labels_list):
    with open(l) as label_file:
        labels = json.load(label_file)
        image_id = l.split('/')[-1].split('.')[0]
        image_ids.append(image_id)

        for b in labels['frames'][0]['objects']:
            if 'box2d' in b:
                obj_dict= { 'bbox': [b['box2d']['x1'], b['box2d']['y1'],
                                     b['box2d']['x2'], b['box2d']['y2']],
                            'name': image_id,
                            'category': b['category']
                          }
                annotations.append(obj_dict)

                if b['category'] not in classes:
                    classes.append(b['category'])

bdd_annotations = { 'image_ids' : image_ids,
                    'classes' : classes,
                    'annotations' : annotations
                  }
print(classes)
with open('bdd_train.json', 'w') as fp:
    json.dump(bdd_annotations, fp)
