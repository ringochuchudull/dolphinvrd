from __future__ import absolute_import, division, print_function
from helpers.parser import DolphinParser
from helpers.utility import PrintColour, ObjColour

import os
import glob
import json

import torch
import torch.utils.data

from PIL import Image

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

if __name__ == '__main__':
    pass