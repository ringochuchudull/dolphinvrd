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
from model.helper.parser import GeneralParser

# TODO for Winson/Derek: Edit this class
# Google Pytorch dataset for visual genome if neccessarily
def MiniVGLoader(GeneralDataset):

    def __init__(self, data_path, transforms=None,):
        super(GeneralDataset, self).__init__()

if __name__ == '__main__':

    parse = GeneralParser()
    parse_options = parse.parse()

    minivg = MiniVGLoader(data_path=parse_options.data_path)