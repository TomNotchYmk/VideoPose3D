import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import cv2
import time
import argparse
import sys
import os
import glob

def read_camera():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        yield frame

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)

boxes = []
segments = []
keypoints = []

for frame_i, im in enumerate(read_camera()):
    t = time.time()
    outputs = predictor(im)['instances'].to('cpu')

    print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

    has_bbox = False
    if outputs.has('pred_boxes'):
        bbox_tensor = outputs.pred_boxes.tensor.numpy()
        if len(bbox_tensor) > 0:
            has_bbox = True
            scores = outputs.scores.cpu().numpy()[:, None]
            bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
    if has_bbox:
        kps = outputs.pred_keypoints.numpy()
        print(kps)
        kps_xy = kps[:, :, :2]
        kps_prob = kps[:, :, 2:3]
        kps_logit = np.zeros_like(kps_prob) # Dummy
        kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
        kps = kps.transpose(0, 2, 1)
    else:
        kps = []
        bbox_tensor = []

    # Mimic Detectron1 format
    cls_boxes = [[], bbox_tensor]
    cls_keyps = [[], kps]

    boxes.append(cls_boxes)
    segments.append(None)
    keypoints.append(cls_keyps)

    # Video resolution
    metadata = {'w': im.shape[1], 'h': im.shape[0],}