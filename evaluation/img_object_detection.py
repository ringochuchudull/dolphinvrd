'''
Author: Derek ka kin Lau, Ringo Chu
Usage:

1. Set "export PYTHONPATH=$PWD" at root folder, the folder that has .git
2. At root folder

    python evaluation\img_object_detection.py \
                        --data_path <DATA> \
                        --model_path <FOLDER>\<PARAM>.pth --device cpu

3. Any questions go to Ringo, do not bother to ask
'''

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.helper.parser import GeneralParser
from dataset.vidvrddataset import VideoVRDDataset, ObjectDetectVidVRDDataset

import model.helper.vision.transforms as T
import model.helper.vision.utils as util
from model.helper.vision.engine import train_one_epoch
from model.helper.utility import cpu_or_gpu

def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.40
    return model


# TODO: Derek and Winson Please get it Done
# Description: Single Image Detection
def evaluate_COCO(dataloader, model):
    pass

def main(parse_options):
    testset_detection = ObjectDetectVidVRDDataset(data_path=parse_options.data_path,
                                                   set='test',
                                                   transforms=None)

    device = parse_options.device
    checkpoint = torch.load(parse_options.model_path, map_location=cpu_or_gpu(device))

    fasterrcnn_resnet50_fpn = get_detection_model(testset_detection.get_num_classes()+1)
    fasterrcnn_resnet50_fpn.load_state_dict(checkpoint['model_state_dict'])

    evaluate_COCO(testset_detection, fasterrcnn_resnet50_fpn)

if __name__ =='__main__':

    parse = GeneralParser()
    parse_options = parse.parse()

    main(parse_options)
