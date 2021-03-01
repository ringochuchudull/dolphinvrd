'''
Modified from: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
'''
from __future__ import print_function, division

import os, glob, json
from torchvision import transforms as T
import torch
from PIL import Image
import numpy as np
import cv2

from .generalloader import GeneralLoader

class VideoVRDLoader(GeneralLoader):

    def __init__(self, data_path,
                 set='train',
                 frames_per_segment: int = 5,
                 imagefile_template: str='{:06d}.jpg',
                 transforms=None,
                 _vis_threshold=0.2):

        super(GeneralLoader, self).__init__()

        self._frame_per_segment = frames_per_segment
        self.imagefile_template = imagefile_template

        self.video_names = []       # Name [name1, name2 ... name_n]
        self._videos_frames = []    # Video frames path [ [video1 frames1]... [vidoo frames n] ]
        self._bbs_info = []
        self._classes = []
        self._classes_info = []
        self._relation_instace = []

        self.transforms = transforms

        # # # # # # # # # # #
        # Start processing# #
        # # # # # # # # # # #

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


            ### This is where the problem is lying!!!!!

            print(f'Frame Count {info["frame_count"]} ')
            print(f'frame path Count {len(video_frame_path)}')
            print(f'Traj Count {len(info["trajectories"])}')

            print(info["trajectories"])

            input()

            # Only keep the frames with Boxes
            video_frame_path = video_frame_path[0:len(info['trajectories'])]

            assert len(info["trajectories"]) == len(video_frame_path), "Number of trajectories should equal to video frame path"

            this_objid = {so["tid"]:so["category"] for so in info["subject/objects"]}
            _bb, _cls = [], []
            _rel = [[] for _ in range(len(info['trajectories']))]

            for idx, traj in enumerate(info["trajectories"]):
                if np.equal(len(traj), 0):
                    _bb.append([-1, -1, -1, -1])
                    _cls.append(-1)
                else:
                    _bbsub = []
                    for t in traj:
                        _bbsub.append([t["bbox"]["xmin"], t["bbox"]["ymin"], t["bbox"]["xmax"], t["bbox"]["ymax"]])
                        _cls.append(this_objid[t["tid"]])
                    _bb.append(_bbsub)

                for r in info["relation_instances"]:
                    if r['begin_fid'] <= idx <= r['end_fid']:
                        _rel[idx].append([r['subject_tid'], r['predicate'], r['object_tid']])

            print(len(_bb), len(video_frame_path))
            assert len(_bb)==len(video_frame_path), "Each frame should has at least a bounding box"

            self.video_names.append(video_id)
            self._videos_frames.append(video_frame_path)
            self._bbs_info.append(_bb)
            self._classes.append(_cls)
            self._classes_info.append(this_objid)
            self._relation_instace.append(_rel)

            # assert len(self._relation_instace) == len(self._bbs_info) == len(self._classes) == len(self._classes_info)

        self._vis_threshold = _vis_threshold
        self.transforms = transforms

    @property
    def num_classes(self):
        return len(self._classes)

    def _load_frames(self, frames):
        return [Image.open(f) for f in frames]

    def __getitem__(self, idx):

        record = self._load_frames(self._videos_frames[idx])
        bbox = self._bbs_info[idx]
        cls = self._classes[idx]
        clsinfo = self._classes_info[idx]
        relins = self._relation_instace[idx]

        # Covert to Files
        ImtoTensor = ImglistToTensor()
        record = ImtoTensor(record)

        if self.transforms is not None:
            transformsList = [t for t in self.transforms]
            transformFunc = T.Compose(transformsList)

        blob = {"record": record, "bbox": bbox, "cls": cls, "clsinfo": clsinfo, "relins": relins}
        return blob

    def __len__(self):
        return len(self.video_names)

    def __str__(self):
        return f'This is VideoVRD loader of length {self.__len__()}'

    def visualise(self, index, display=False):
        print(f'Visualising video {index}')
        blob = self.__getitem__(index)

        video = blob['record']
        for frame in video:
            frame = frame.numpy().transpose(1, 2, 0)
            frame = frame[:, :, ::-1]
            cv2.imshow('Color image', frame)
            cv2.waitKey(5)

        bbox = blob['bbox']

        print(len(bbox), len(video))
        for b in bbox:
            print(b.shape)

        return None

class ImglistToTensor(torch.nn.Module):
    """
    Converts a list of PIL images in the range [0,255] to a torch.FloatTensor
    of shape (NUM_IMAGES x CHANNELS x HEIGHT x WIDTH) in the range [0,1].
    Can be used as first transform for ``VideoFrameDataset``.
    """
    def forward(self, img_list):
        """
        Converts each PIL image in a list to
        a torch Tensor and stacks them into
        a single tensor.
        Args:
            img_list: list of PIL images.
        Returns:
            tensor of size ``NUM_IMAGES x CHANNELS x HEIGHT x WIDTH``
        """
        return torch.stack([T.functional.to_tensor(pic) for pic in img_list])