# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List

from mmengine.dataset import BaseDataset
import tempfile
from mmrotate.registry import DATASETS
from .dota import DOTADataset
import os
import os.path as osp
import re
import shutil
import tempfile
import time
from venv import create
import mmcv
import numpy as np
import torch

from PIL import Image

@DATASETS.register_module()
class FAIR1M_Dataset(DOTADataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    # CLASSES = ('Passenger-Ship', 'Motorboat', 'Fishing-Boat',
    #            'Tugboat', 'other-ship', 'Engineering-Ship', 'Liquid-Cargo-Ship',
    #            'Dry-Cargo-Ship', 'Warship', 'Small-Car', 'Bus',
    #            'Cargo-Truck', 'Dump-Truck', 'other-vehicle', 'Van',
    #            'Trailer', 'Tractor', 'Excavator', 'Truck-Tractor',
    #            'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
    #            'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',
    #            'other-airplane', 'Baseball-Field', 'Basketball-Court',
    #            'Football-Field', 'Tennis-Court', 'Roundabout', 'Intersection', 'Bridge')
    CLASSES = ('Passenger', 'Motorboat', 'Fishing',#2
               'Tugboat', 'other-ship', 'Engineering', 'Liquid',#6
               'Dry', 'Warship', 'Small', 'Bus',#10
               'Cargo', 'Dump', 'other-vehicle', 'Van',#14
               'Trailer', 'Tractor', 'Excavator', 'Truck',#18
               'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',#22
               'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',#28
               'other-airplane', 'Baseball', 'Basketball',#31
               'Football', 'Tennis', 'Roundabout', 'Intersection', 'Bridge')
    
    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
         (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
         (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
         (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
         (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
         (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
         (92, 0, 73), (133, 129, 255), (78, 180, 255),(209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164)]
    
    metainfo = {'classes':CLASSES,'palette':PALETTE}
    METAINFO = {'classes':CLASSES,'palette':PALETTE}

