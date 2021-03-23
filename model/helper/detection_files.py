'''
Author: Ringo S W Chu, Peter Hohin Lee, Winson Luk
'''

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from model.helper.parser import GeneralParser # Comment Line Arguement Parser
from dataset.vidvrddataset import VideoVRDDataset, ObjectDetectVidVRDDataset

import model.helper.vision.transforms as T
import model.helper.vision.utils as util
from model.helper.vision.engine import train_one_epoch, evaluate
from model.helper.utility import cpu_or_gpu

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
    # transforms.append(T.ToTensor())
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

def train(arguement):
    trainset_detection = ObjectDetectVidVRDDataset(data_path=arguement.data_path,
                                                   set='train',
                                                   transforms=[T.RandomHorizontalFlip(0.5)])

    print(f' Length of the training loader {len(trainset_detection)}')

    data_loader = torch.utils.data.DataLoader(trainset_detection,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=util.collate_fn
                                              )

    device = cpu_or_gpu(arguement.device) # torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    num_classes = trainset_detection.get_num_classes() + 1
    model = get_detection_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 11
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader, device=device)

        if epoch+1 % 5 == 0:
            every_parameter = {'epoch': epoch,
                              'model_state_dict': model.state_dict(),
                              'optimizer_state_dict': optimizer.state_dict()
                              }
            torch.save(every_parameter, os.path.join(arguement.model_path, f"vidvrd_detector_{epoch}.pth"))

def test():
    testset_detection = ObjectDetectVidVRDDataset(data_path=parse_options.data_path,
                                                   set='test',
                                                   transforms=None)

    data_loader = torch.utils.data.DataLoader(testset_detection,
                                              batch_size=2,
                                              shuffle=True,
                                              num_workers=4,
                                              collate_fn=util.collate_fn
                                              )


if __name__ == '__main__':
    parse = GeneralParser()
    parse_options = parse.parse()

    trainset_detection = VideoVRDDataset(data_path=parse_options.data_path,
                                               set='train')
    '''
    for i in range(200,201):
        trainset_detection.visualise(i)
    '''
    train(parse_options)

