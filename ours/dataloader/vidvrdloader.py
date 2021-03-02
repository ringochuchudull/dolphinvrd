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
        self._bbs_info = []         # Bounding Boxes [ [] ... [] ]
        self._classes = []          #
        self._classes_info = []     #
        self._relation_instace = [] #

        self._heights, self._widths = [], []

        self.ImglistToTensor = ImglistToTensor()
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

            self._heights.append(info['height'])
            self._widths.append(info['width'])

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

            assert len(_bb)==len(video_frame_path), "Each frame should has at least a bounding box"

            self.video_names.append(video_id)
            self._videos_frames.append(video_frame_path)
            self._bbs_info.append(_bb)
            self._classes.append(_cls)
            self._classes_info.append(this_objid)
            self._relation_instace.append(_rel)

        # assert len(self._relation_instace) == len(self._bbs_info) == len(self._classes) == len(self._classes_info)
        assert len(self.video_names) == len(self._videos_frames) == len(self._bbs_info) == len(self._classes) == len(self._classes_info)
        assert len(self.video_names) == len(self._heights) == len(self._widths)

        self._vis_threshold = _vis_threshold
        self.transforms = transforms

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):

        record = self._videos_frames[idx]
        bbox = self._bbs_info[idx]
        cls = self._classes[idx]
        clsinfo = self._classes_info[idx]
        relins = self._relation_instace[idx]
        height = self._heights[idx]
        width = self._widths[idx]

        blob = {"record": record,
                "bbox": bbox,
                "cls": cls,
                "clsinfo": clsinfo,
                "relins": relins,
                "height":height,
                "width":width}

        return blob

    def _load_frames(self, frames):
        if len(frames) == 1:  # Special Case for a single frame
            return [Image.open(frames[0])]
        return [Image.open(f) for f in frames]

    def gives_stack_of_frames(self, frames):
        frames = self._load_frames(frames)
        frames = self.ImglistToTensor(frames)
        return frames

    def visualise(self, index, display=False):
        print(f'Visualising video {index}')
        blob = self.__getitem__(index)

        video = blob['record']
        bbox = blob['bbox']

        # Resizing the frame first
        height, width = blob['height'], blob['width']

        boundary = 0
        for frame in video:
            frame = self.gives_stack_of_frames([frame])[0]
            frame = frame.numpy().transpose(1, 2, 0)
            frame = frame[:, :, ::-1]
            # Resize
            ratio = 720.0 / height
            size = int(round(width * ratio)) + 2 * boundary, int(round(height * ratio)) + 2 * boundary
            frame = cv2.resize(frame, (size[0]-2*boundary, size[1]-2*boundary))

            # Pick up from here
            for b in bbox:
                print(b)
                input()

            cv2.imshow('Color image', frame)
            cv2.waitKey(2)

        cv2.destroyAllWindows()
        return None

    def get_segment_size(self):
        return self._frame_per_segment

    def __len__(self):
        return len(self.video_names)

    def __str__(self):
        return f'This is VideoVRD loader of length {self.__len__()}'



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