#! /usr/bin/env python

import os
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes, makedirs
from utils.bbox import draw_boxes
from keras.models import load_model
from tqdm import tqdm
import numpy as np


def loadModel(config):


    ###############################
    #   Load the model
    ###############################
    os.environ['CUDA_VISIBLE_DEVICES'] = config['train']['gpus']
    infer_model = load_model(config['train']['saved_weights_name'])

    return infer_model


def detect_fonts(gray, config, infer_model, standBox):

    # predict the bounding boxes
    boxes = get_yolo_boxes(infer_model, [gray], 416, 416, config['model']['anchors'], 0.5, 0.45)[0]

    new_boxes = []
    x1 = standBox[0]
    y1 = standBox[1]
    x2 = standBox[2]
    y2 = standBox[3]
    for box in boxes:
        box.xmin += x1
        box.ymin += y1
        box.xmax += x1
        box.ymax += y1
        new_boxes.append(box)

    #rImg = draw_boxes(image, new_boxes, config['model']['labels'], 0.5)

    return new_boxes
