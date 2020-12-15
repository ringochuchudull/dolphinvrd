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
from helpers.parser import DolphinParser

import os
import glob
import json

import torch
import torch.utils.data

from PIL import Image

import numpy as np
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import .vision.transforms as T
from .vision.engine import train_one_epoch, evaluate
import .vision.utils as utils


class DOLPHIN(torch.utils.data.Dataset):

    def __init__(self, data_path, set, mode='general', vis_threshold=0.2, transforms=None):

        # load all image files, sorting them to
        # ensure that they are aligned
        self._img_paths = []
        self._bbs_info = []

        self.root = data_path
        self._img_container = 'images' if set is 'Train' else 'images2'

        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"

        path_to_videos = os.path.join(self.root, set, '*')
        self.videos = list(sorted(glob.glob(path_to_videos)))
        assert len(self.videos) != 0

        for video_path in self.videos:

            # read configure.JSON
            with open(os.path.join(video_path, 'configuration.json'), 'r') as l:
                labels = l.read()
            labels = json.loads(labels)

            # A list of images path
            imgpaths = sorted(glob.glob(os.path.join(video_path, self._img_container, '*')))
            _temporary_bbs = []

            for frame in imgpaths:
                _, frame_name = os.path.split(frame)
                _temporary_bbs.append(labels[frame_name])

            assert len(imgpaths) == len(_temporary_bbs)
            self._img_paths += imgpaths
            self._bbs_info += _temporary_bbs

        assert len(self._img_paths) == len(self._bbs_info)

        if mode.lower() == 'specific':
            self._classes = ('background', 'd1', 'd2', 'd3', 'd4', 'pipe')
        else:
            self._classes = ('background', 'dolphin', 'pipe')

        self._vis_threshold = vis_threshold
        self.transforms = transforms

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):

        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        bbinfos = self._bbs_info[idx]
        num_objs = len(bbinfos)

        boxes = []
        labels = []

        for i in range(num_objs):

            this_config = bbinfos[i]
            # Sort out the bounding boxes first
            tl, tr = this_config['Bounding Box left'], this_config['Bounding Box top']
            height, width = this_config['Bounding box height'], this_config['Bounding box width']
            xmin, ymin, xmax, ymax = tl, tr - height, tl + width, tr
            boxes.append([xmin, ymin, xmax, ymax])

            # Sort out the label after
            this_id = this_config['Identity']

            if len(self._classes) == 3:  # General Mode
                if this_id in ['Angelo', 'Toto', 'Anson', 'Ginsan']:
                    labels.append(1)
                elif this_id in ['Pipe']:
                    labels.append(2)
                else:
                    pass

            elif len(self._classes) == 6:  # Specific Mode

                if this_id is 'Angelo':
                    labels.append(1)
                elif this_id is 'Toto':
                    labels.append(2)
                elif this_id is 'Anson':
                    labels.append(3)
                elif this_id is 'Ginsan':
                    labels.append(4)
                elif this_id is 'Pipe':
                    labels.append(5)
                else:
                    pass
            else:
                raise Exception  # Error

        assert len(boxes) == len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return 'This is DOLPHIN DETECTION LOADER'


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

    model.roi_heads.nms_thresh = 0.42

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
    dp = DolphinParser()
    dp_options = dp.parse()

    dataset = DOLPHIN(data_path=dp_options.data_path,
                      set='Train',
                      mode='general',
                      transforms=get_transform(train=True))
    dataset_test = DOLPHIN(data_path=dp_options.data_path,
                           set='Test',
                           mode='general',
                           transforms=get_transform(train=False))

    print(f'Length of Train: {len(dataset)}')
    print(f'Length of Test: {len(dataset_test)}')

    '''
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    '''

    '''Debug
    print(dataset[123])
    input()
    print(dataset_test[600])
    input()
    '''

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    print(f'Length of Train: {len(data_loader)}')
    print(f'Length of Test: {len(data_loader_test)}')

    device = None
    if (dp_options.device.lower() in ['cuda', 'gpu']) and torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # if mode = general then 3, if mode = specific then 5
    num_classes = 3
    # get the model using our helper function
    # model = get_instance_segmentation_model_v2(num_classes)
    model = get_detection_model(num_classes)
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 51
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset

        if epoch % 3 == 0:
            evaluate(model, data_loader_test, device=device)
            every_parameter = {'epoch': epoch,
                               'model_state_dict': model.state_dict(),
                               'optimizer_state_dict': optimizer.state_dict()
                               }
            torch.save(every_parameter, os.path.join("models", f"general_detector_{epoch}.pth"))