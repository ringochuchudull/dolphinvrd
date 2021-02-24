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

