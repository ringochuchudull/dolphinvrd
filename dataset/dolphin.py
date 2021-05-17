from __future__ import absolute_import, division, print_function
from model.helper.parser import DolphinParser

import os
import glob
import json

import torch
import torch.utils.data


from PIL import Image

class DOLPHIN(torch.utils.data.Dataset):

    def __init__(self, data_path, set, mode='general', vis_threshold=0.2, segment_size=30, transforms=None):

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

        self.window_size = segment_size 

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):

        return self.src_one_idx(idx)
    
    def src_one_idx(self, idx):

        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")

        bbinfos = self._bbs_info[idx]
        num_objs = len(bbinfos)

        boxes = []
        labels = []

        behaviour = []
        for i in range(num_objs):

            this_config = bbinfos[i]
            # 
            tl, tr = this_config['Bounding Box left'], this_config['Bounding Box top']
            height, width = this_config['Bounding box height'], this_config['Bounding box width']
            xmin, ymin, xmax, ymax = tl, tr - height, tl + width, tr
            boxes.append([xmin, ymin, xmax, ymax])
    
            # 
            this_id = this_config['Identity']

            if len(self._classes) == 3:  # General Mode
                if this_id in ['Angelo', 'Toto', 'Anson', 'Ginsan']:
                    labels.append(1)
                elif this_id in ['Pipe']:
                    labels.append(2)
                else:
                    pass

            elif len(self._classes) == 6:  # Specific Mode

                if this_id == 'Angelo':
                    labels.append(1)
                elif this_id == 'Toto':
                    labels.append(2)
                elif this_id == 'Anson':
                    labels.append(3)
                elif this_id == 'Ginsan':
                    labels.append(4)
                elif this_id == 'Pipe':
                    labels.append(5)
                else:
                    pass
            else:
                raise Exception  # Error

            # Inv, Obs, Moving, Eat, Coop, Int
            _action = this_config['Class']['Behaviour']
            position = {'Coop':0, 'Eat':1, 'Int':2, 'Inv':3, 'Moving':4, 'Obs':5, 'Tug':6, 'Following':7, 'None':8}
            one_hot_behav = [0, 0, 0, 0, 0, 0 ,0 ,0 ,0]
            if len(_action) == 0:
                one_hot_behav[position['None']] = 1
            else:
                one_hot_behav[position[_action]] = 1
            behaviour.append(one_hot_behav)


        assert len(boxes) == len(labels)
        assert len(behaviour) == len(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
                
        behaviour = torch.as_tensor(behaviour, dtype=torch.int64)

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
        target["behaviour"] = behaviour

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["img"] = img
        return img, target

    def __len__(self):
        return len(self._img_paths)

    def __str__(self):
        return 'This is DOLPHIN DETECTION LOADER'

from collections import defaultdict

class DOLPHINVIDEOVRD(DOLPHIN):

    def __getitem__(self, idx):        
        clip = []
        for i in range(idx, idx+self.window_size):
            _, t = self.src_one_idx(i)
            clip.append(t)    

        track_id = self.batchify_single(clip)
        return clip, track_id
    
    def batchify_single(self, clip):    
        track_id = {i:{'traj':torch.zeros((self.window_size, 4)), 'motion':None} for i in range(1,6)}
    
        assert len(clip) == self.window_size
        for i in range(len(clip)):        
            for j, (bb, dolpid, behav) in enumerate(zip(clip[i]['boxes'], clip[i]['labels'], clip[i]['behaviour'])):
                track_id[dolpid.item()]['traj'][i, :] = bb
            
                if i == self.window_size-1:
                    track_id[dolpid.item()]['motion'] = behav

        return track_id
            
    def __len__(self):
        return len(self._img_paths) - self.window_size

    def __str__(self):
        return 'This is DOLPHIN VRD Video Loader'

    def __get_window_size__(self):
        return self.window_size

import model.helper.vision.transforms as T

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # https://pytorch.org/docs/stable/torchvision/transforms.html
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)



if __name__ == '__main__':
    check = DOLPHINVIDEOVRD('../DOLPHIN/', set='Train', transforms=get_transform(False))
    data_loader = torch.utils.data.DataLoader(check, batch_size=1, shuffle=False)

