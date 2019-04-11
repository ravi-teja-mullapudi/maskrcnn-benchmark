import torch
import torch.utils.data as data
from PIL import Image
import os
import json
from collections import defaultdict
from maskrcnn_benchmark.structures.bounding_box import BoxList

def group_by_key(detections, key):
    groups = defaultdict(list)
    for d in detections:
        groups[d[key]].append(d)
    return groups

class BDDDataset(data.Dataset):

    CLASSES = (
        "__background__",
        "traffic light",
        "traffic sign",
        "car",
        "person",
        "bus",
        "truck",
        "rider",
        "bike",
        "motor",
        "train",
    )

    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        assert(ann_file)
        with open(ann_file, 'r') as label_file:
            self.annotations = json.load(label_file)
        self.ids = self.annotations['image_ids']
        self.transforms = transforms
        self.ann_ids = group_by_key(self.annotations['annotations'], 'name')

        cls = BDDDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        detections = self.ann_ids[img_id]

        boxes = [ det['bbox'] for det in detections ]
        classes = [ det['category'] for det in detections ]
        classes = [ self.class_to_ind[c] for c in classes ]
        classes = torch.tensor(classes)

        img_path = os.path.join(self.img_dir, img_id + '.jpg')
        img = Image.open(img_path).convert('RGB')

        target = BoxList(boxes, img.size, mode='xyxy')
        target.add_field("labels", classes)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def __len__(self):
        return len(self.ids)

    def get_img_info(self, idx):
        return {"height": 720, "width": 1280}
