import os
import glob
import torch
import json
from PIL import Image

import ours.helper.vision.transforms as T
from ours.helper.vision.engine import train_one_epoch, evaluate
import ours.helper.vision.utils as utils

class GeneralLoader(torch.utils.data.Dataset):

    def __init__(self, data_path, set, transforms=None, _vis_threshold=0.2):

        # load all image files, sorting them to
        # ensure that they are aligned
        self._img_paths = []
        self._bbs_info = []

        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"
    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):
        return None

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return 'This is General Detection LOADER'

class VideoVRDParser(GeneralLoader):

    def __init__(self, data_path, set, transforms=None, _vis_threshold=0.2):
        super(GeneralLoader, self).__init__()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.video_names = []
        self._videos_frames = []
        self._bbs_info = []
        self.classes = []

        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"
        path_to_json = os.path.join(self.root, set, '*.json')

        self.all_json_anno = list(sorted(glob.glob(path_to_json)))
        assert len(self.all_json_anno) != 0

        for idx, js in enumerate(self.all_json_anno):
            with open(js, 'r') as l:
                info = l.read()
            info = json.loads(info)

            video_id = info['video_id']

            video_frame_path = glob.glob(os.path.join(self.root, set, video_id, '*.jpeg'))
            assert len(video_frame_path) == info['frame_count'], 'Frame Count and actual count mismatch '

            # Only keep the frames with Boxes
            temp_traj = info['trajectories']
            video_frame_path = video_frame_path[0:len(temp_traj)]

            self.video_names.append(video_id)
            self._videos_frames.append(video_frame_path)
            self._bbs_info.append(info['trajectories'])

            '''
            # Number of frames
            print(f'Number of frames: {info["frame_count"]}')
            # Object Detector
            print(f'Length of trajectories: {len(info["trajectories"])}')
            # Relations
            print(f'Relation Instances:\n {info["relation_instances"]}')
            # Subject/Object Class
            print(f'Subject Object: \n {info["subject/objects"]}')
            '''

        self._vis_threshold = _vis_threshold
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
        return 'This is General Detection LOADER'


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
