'''
Modified from: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
'''

import os
import glob
from torchvision import transforms
import torch
import json
from PIL import Image
import numpy as np


from .generalloader import GeneralLoader

class VideoVRDLoader(GeneralLoader):

    def __init__(self, data_path,
                 frames_per_segment: int = 1,
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
        #self.total_classes = []

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
            video_frame_path = video_frame_path[0:len(info['trajectories'])]

            # One Video
            this_objid = {so["tid"]:so["category"] for so in info["subject/objects"]}
            _bb, _cls = [], []
            _rel = [[] for _ in range(len(info['trajectories']))]

            for idx, traj in enumerate(info["trajectories"]):

                if np.equal(len(traj), 0):
                    _bb.append([-1, -1, -1, -1])
                    _cls.append(-1)

                for t in traj:
                    _bb.append([t["bbox"]["xmin"], t["bbox"]["ymin"], t["bbox"]["xmax"], t["bbox"]["ymax"]])
                    _cls.append(this_objid[t["tid"]])

                for r in info["relation_instances"]:
                    if r['begin_fid'] <= idx <= r['end_fid']:
                        _rel[idx].append([r['subject_tid'], r['predicate'], r['object_tid']])

            self.video_names.append(video_id)
            self._videos_frames.append(video_frame_path)
            self._bbs_info.append(_bb)
            self._classes.append(_cls)
            self._classes_info.append(this_objid)
            self._relation_instace.append(_rel)
            assert len(self._relation_instace) == len(self._bbs_info) == len(self._classes) == len(self._classes_info)

        self._vis_threshold = _vis_threshold
        self.transforms = transforms

    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):


        return None

    def __len__(self):
        return len(self.video_names)

    def __str__(self):
        return 'This is VideoVRD loader'


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
        return torch.stack([transforms.functional.to_tensor(pic) for pic in img_list])