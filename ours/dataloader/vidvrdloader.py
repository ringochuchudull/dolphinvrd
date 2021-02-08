import os
import glob
import torch
import json
from PIL import Image
import numpy as np

from .generalloader import GeneralLoader

class VideoVRDLoader(GeneralLoader):

    def __init__(self, data_path, set, transforms=None, _vis_threshold=0.2):
        super(GeneralLoader, self).__init__()
        # load all image files, sorting them to
        # ensure that they are aligned
        self.video_names = []
        self._videos_frames = []
        self._bbs_info = []
        self._classes = []

        self.total_classes = []

        self.root = data_path
        assert os.path.exists(self.root), "Your -->  --data_path <-- got problem"
        path_to_json = os.path.join(self.root, set, '*.json')

        self.all_json_anno = list(sorted(glob.glob(path_to_json)))
        assert len(self.all_json_anno) != 0

        for idx, js in enumerate(self.all_json_anno):
            print(idx)
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
            print(objid); input()
            # Number of frames
            print(f'Number of frames: {info["frame_count"]}')
            # Object Detector
            print(f'Length of trajectories: and {len(info["trajectories"])}')
            print(f'Length of trajectories: and ')
            '''

            # One Video

            this_objid = { so["tid"]:so["category"] for so in info["subject/objects"]}

            _bb, _cls, _rel = [], [], []
            for idx, traj in enumerate(info["trajectories"]):

                if np.equal(len(traj), 0):
                    _bb.append([-1, -1, -1, -1])
                    _cls.append(-1)

                for t in traj:
                    _bb.append([t["bbox"]["xmin"], t["bbox"]["ymin"], t["bbox"]["xmax"], t["bbox"]["ymax"]])
                    _cls.append(this_objid[t["tid"]])

            self._bbs_info.append(_bb)
            self._classes.append(_cls)

            # Relations
            print(f'Relation Instances:\n {info["relation_instances"]}')
            # Subject/Object Class
            print(f'Subject Object: \n {info["subject/objects"]}')

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