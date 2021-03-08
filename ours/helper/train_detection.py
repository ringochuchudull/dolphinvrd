'''
Author: Ringo S W Chu, Peter Hohin Lee, Winson Luk
Prerequisite:
    1. Put these two lines in your terminal/command prompt
    pip install cython
    pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    2. Download TorchVision repo to use some files from references/detection
    git clone https://github.com/pytorch/vision.git
    cd vision
    git checkout v0.3.0
    cp references/detection/utils.py ../
    cp references/detection/transforms.py ../
    cp references/detection/coco_eval.py ../
    cp references/detection/engine.py ../
    cp references/detection/coco_utils.py ../
    2. Referencing Material
    Revised from here: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
Note:
    1. Geometric clarification
    Your rectangle Box with top-left, top-right
    For example:
    idno = bb_info['Identity']
    tl, tr = np.rint(bb_info['Bounding Box left']), np.rint(bb_info['Bounding Box top'])
    tl, tr = tl.astype(int), tr.astype(int)
    height, width = np.rint(bb_info['Bounding box height']), np.rint(bb_info['Bounding box width'])
    height, width = height.astype(int), width.astype(int)
    p1, p2 = (tl, tr-height), (tl+width, tr)
'''

from __future__ import absolute_import, division, print_function
import os, glob, json

from PIL import Image
import numpy as np
import cv2

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ours.helper.parser import GeneralParser
from ours.dataloader.vidvrdloader import VideoVRDLoader
from ours.dataloader.transformfunc import ImglistToTensor


import ours.helper.vision.transforms as T
from ours.helper.vision.engine import train_one_epoch, evaluate
import ours.helper.vision.utils as utils



def get_instance_segmentation_model_v2(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.nms_thresh = 0.45

    return model


def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.roi_heads.nms_thresh = 0.45

    return model


# torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)


def evaluate_and_write_result_files(model, data_loader):
    model.eval()
    results = {}
    for imgs, targets in data_loader:
        imgs = [img.to(device) for img in imgs]

        with torch.no_grad():
            preds = model(imgs)

        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(),
                                                  'scores': pred['scores'].cpu()}

if __name__ == '__main__':
    parse = GeneralParser()
    parse_options = parse.parse()

    train_set = VideoVRDLoader(data_path=parse_options.data_path,
                               set='train',
                               transforms=None)

    for idx in range(173, 300):
        train_set.visualise(idx,
                            draw_box=True,
                            draw_relation=True)
