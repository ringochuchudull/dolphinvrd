'''
Description: Simple Loader for a Visual Genome 20K
Author = 'Ringo Chu'
'''

from __future__ import print_function, division

import os, glob, json
from PIL import Image
import numpy as np
import cv2

from .generaldataset import GeneralLoader

def MiniVGLoader(GeneralLoader):

    def __init__(self):
        pass