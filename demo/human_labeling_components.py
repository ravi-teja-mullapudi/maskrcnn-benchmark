from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import json

import numpy as np
import tensorflow as tf
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
            options=['Yes', 'No'],
            description='Positive example?',
            button_style='',
        )
        label_widgets.append(label_widget)
        hbox = widgets.HBox([image_widget, mask_widget])
        vbox = widgets.VBox([hbox, label_widget])
        vboxes.append(vbox)

    return widgets.VBox(vboxes), label_widgets
