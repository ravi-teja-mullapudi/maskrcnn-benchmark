import json
import cv2
import os
import numpy as np
from shutil import copyfile

classes = ['traffic light', 'traffic sign', 'person', 'truck', 'car', 'bus', 'rider', 'motor', 'bike', 'train']

def draw_boxes(img, labels):
    img = img.copy()
    for l in labels:
        bbox = np.array(l['bbox']).astype(int)
        top_left, bottom_right = bbox[:2].tolist(), bbox[2:].tolist()
        img = cv2.rectangle(img, tuple(top_left), tuple(bottom_right),
                            (0, 255, 0), 3)
        cv2.putText(img, l['category'], tuple(top_left), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 0, 0), 2)
    return img


if __name__ == "__main__":
    image_dir = '/n/pana/scratch/ravi/bdd/bdd100k/images/100k/val/'
    label_dir = '/n/pana/scratch/ravi/bdd/bdd100k/labels/100k/val'
    output_dir = '/n/pana/scratch/ravi/bdd_vis_results_no_night'
    video_dir = '/n/pana/scratch/ravi/bdd/bdd100k/videos/100k/val/'

    for cls in classes:
        fp = open('bdd_analysis_pretrained.json', 'r')
        analysis = json.load(fp)

        fp = open('pretrained.json', 'r')
        pretrained_preds = json.load(fp)

        fp = open('supervised.json', 'r')
        supervised_preds = json.load(fp)

        filtered_analysis = {}
        supervised_labels = {}
        pretrained_labels = {}

        for l in pretrained_preds:
            if l['score'] > 0.5 and l['category'] == cls:
                if l['name'] in pretrained_labels:
                    pretrained_labels[l['name']].append(l)
                else:
                    pretrained_labels[l['name']] = [l]

        for l in supervised_preds:
            if l['score'] > 0.5 and l['category'] == cls:
                if l['name'] in supervised_labels:
                    supervised_labels[l['name']].append(l)
                else:
                    supervised_labels[l['name']] = [l]

        for img in analysis:
            labels_path = os.path.join(label_dir, img + '.json')
            label_file = open(labels_path)
            gt = json.load(label_file)
            if cls in analysis[img] and gt['attributes']['timeofday'] != 'night':
                filtered_analysis[img] = analysis[img]

        s = sorted(filtered_analysis.items(), key=lambda x: float(x[1][cls][0]))
        print(cls, len(s))

        for im_id in range(min(len(s), 50)):
            image_name = s[im_id][0] + '.jpg'
            image_path = os.path.join(image_dir, image_name)
            img = cv2.imread(image_path)

            labels_path = os.path.join(label_dir, s[im_id][0] + '.json')
            label_file = open(labels_path)
            gt = json.load(label_file)

            gt_labels = []

            for b in gt['frames'][0]['objects']:
                if 'box2d' in b and b['category'] == cls:
                    obj_dict= { 'bbox': [b['box2d']['x1'], b['box2d']['y1'],
                                     b['box2d']['x2'], b['box2d']['y2']],
                                'name': s[im_id][0],
                                'category': b['category']
                              }
                    gt_labels.append(obj_dict)

            if s[im_id][0] not in pretrained_labels:
                pretrained_labels[s[im_id][0]] = []

            if s[im_id][0] not in supervised_labels:
                supervised_labels[s[im_id][0]] = []

            gt_image = draw_boxes(img, gt_labels)
            pretrained_image = draw_boxes(img, pretrained_labels[s[im_id][0]])
            supervised_image = draw_boxes(img, supervised_labels[s[im_id][0]])

            vis_img = np.concatenate((pretrained_image, supervised_image, gt_image), axis=1)
            vis_img_name = s[im_id][0] + '.png'

            vis_img_dir = os.path.join(output_dir, cls)
            if not os.path.exists(vis_img_dir):
                os.makedirs(vis_img_dir)
            vis_img_path = os.path.join(vis_img_dir, vis_img_name)

            video_src_path = os.path.join(video_dir, s[im_id][0] + '.mp4')
            video_des_path = os.path.join(vis_img_dir, s[im_id][0] + '.mp4')
            copyfile(video_src_path, video_des_path)

            print(vis_img_path)
            cv2.imwrite(vis_img_path, vis_img)
