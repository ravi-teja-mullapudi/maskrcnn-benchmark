import torch
import torch.utils.data as data
from PIL import Image
import os
import json
from collections import defaultdict
from maskrcnn_benchmark.structures.bounding_box import BoxList
import glob

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

class OpenImagesDataset(data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        sub_dirs = [ 'train_' + i for i in \
                  [ '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']]

        image_paths = []
        for sd in sub_dirs:
            image_paths = image_paths + glob.glob(os.path.join(img_dir, sd, '*.jpg'))

        image_id_to_path = {}
        for path in image_paths:
            image_id_to_path[path.split('/')[-1].split('.')[0]] = path
        self.image_id_to_path = image_id_to_path

        assert(ann_file)
        with open(ann_file, 'r') as label_file:
            self.annotations = json.load(label_file)
        self.ids = self.annotations['image_ids']
        self.transforms = transforms
        self.ann_ids = group_by_key(self.annotations['annotations'], 'name')

        ann_by_cat = group_by_key(self.annotations['annotations'], 'category')
        cls = ['__background__'] + list(ann_by_cat.keys())
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        detections = self.ann_ids[img_id]

        boxes = [ det['bbox'] for det in detections ]
        classes = [ det['category'] for det in detections ]
        classes = [ self.class_to_ind[c] for c in classes ]
        classes = torch.tensor(classes)

        img_path = self.image_id_to_path[img_id]
        img = Image.open(img_path).convert('RGB')

        width, height = img.size
        scaled_boxes = []
        for box in boxes:
            scaled_box = [ box[0] * width,
                           box[2] * height,
                           box[1] * width,
                           box[3] * height ]
            scaled_boxes.append(scaled_box)

        target = BoxList(scaled_boxes, img.size, mode='xyxy')
        target.add_field("labels", classes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        img_id = self.ids[idx]
        img_path = self.image_id_to_path[img_id]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        return {"height": height, "width": width}
