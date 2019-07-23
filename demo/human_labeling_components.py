from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import numpy as np
import time
import cv2
import random
import math
import multiprocessing as mp
import subprocess
import argparse
import csv

def make_image_level_supervision_widget(images):
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    label_widgets = []
    vboxes = []
    for img, sim_img in images:
        width = 300
        height = int(width * 1.0 / (img.shape[1] / img.shape[0]))
        img = cv2.resize(img, (width, height)).astype(np.uint8)
        _, img_enc = cv2.imencode('.png', img)
        image_widget = widgets.Image(
            value=img_enc.tostring(),
            format='png',
            width=width,
            height=height,
        )
        sim_img = cv2.resize(sim_img, (width, height)).astype(np.uint8)
        _, sim_img_enc = cv2.imencode('.png', sim_img)
        mask_widget = widgets.Image(
            value=sim_img_enc.tostring(),
            format='png',
            width=width,
            height=height,
        )
        label_widget = widgets.ToggleButtons(
            options=['Yes', 'No', 'Unsure'],
            description='Positive example?',
            button_style='',
        )
        label_widgets.append(label_widget)
        hbox = widgets.HBox([image_widget, mask_widget])
        vbox = widgets.VBox([hbox, label_widget])
        vboxes.append(vbox)

    return widgets.VBox(vboxes), label_widgets

def update_query_with_user_feedback(user_positives,
                                    user_negatives,
                                    user_neutrals,
                                    query_similar_embeddings,
                                    current_query):
    current_query = current_query / (np.linalg.norm(current_query, ord=2) + np.finfo(float).eps)
    pos_embeddings = []
    neg_embeddings = []
    neutral_embeddings = []
    for im_id in user_positives:
        embeddings = query_similar_embeddings[im_id]
        pos_embeddings.append(np.mean(embeddings, axis=0))
    for im_id in user_negatives:
        embeddings = query_similar_embeddings[im_id]
        neg_embeddings.append(np.mean(embeddings, axis=0))
    for im_id in neutral_embeddings:
        embeddings = query_similar_embeddings[im_id]
        neutral_embeddings.append(np.mean(embeddings, axis=0))

    pos_average = sum(pos_embeddings)/(len(pos_embeddings) + np.finfo(float).eps)
    neg_average = sum(neg_embeddings)/(len(neg_embeddings) + np.finfo(float).eps)
    refined_query = current_query + (pos_average - neg_average)
    return refined_query
