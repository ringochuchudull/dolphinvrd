from __future__ import print_function, division

import os, glob, json
from PIL import Image
import numpy as np
import cv2

from .transformfunc import ImglistToTensor
from .generalloader import GeneralLoader
from ours.helper.utility import add_bbox, add_straight_line
from ours.helper.utility import generate_random_colour, _COLOR_NAME_TO_RGB

from collections import defaultdict


class VisualGenome(GeneralLoader):

    def __init__(self, data_path,
                 set='train',
                 imagefile_template: str='{:06d}.jpg',
                 transforms=None,
                 _vis_threshold=0.2):

        super(GeneralLoader, self).__init__()

