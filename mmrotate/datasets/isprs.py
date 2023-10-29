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
import zipfile
from collections import defaultdict
from functools import partial

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated


from mmrotate.core import obb2poly_np

from xml.dom.minidom import Document
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

    CLASSES = ('Passenger-Ship', 'Motorboat', 'Fishing-Boat',
               'Tugboat', 'other-ship', 'Engineering-Ship', 'Liquid-Cargo-Ship',
               'Dry-Cargo-Ship', 'Warship', 'Small-Car', 'Bus',
               'Cargo-Truck', 'Dump-Truck', 'other-vehicle', 'Van',
               'Trailer', 'Tractor', 'Excavator', 'Truck-Tractor',
               'Boeing737', 'Boeing747', 'Boeing777', 'Boeing787',
               'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',
               'other-airplane', 'Baseball-Field', 'Basketball-Court',
               'Football-Field', 'Tennis-Court', 'Roundabout', 'Intersection', 'Bridge')
    
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


    def merge_det(self, results, nproc=4):
        """Merging patch bboxes into full image.

        Args:
            results (list): Testing results of the dataset.
            nproc (int): number of process. Default: 4.
        """
        collector = defaultdict(list)
        for idx in range(len(self)):
            result = results[idx]
            img_id = self.img_ids[idx]
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            new_result = []
            for i, dets in enumerate(result):
                bboxes, scores = dets[:, :-1], dets[:, [-1]]
                ori_bboxes = bboxes.copy()
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
                labels = np.zeros((bboxes.shape[0], 1)) + i
                new_result.append(
                    np.concatenate([labels, ori_bboxes, scores], axis=1))

            new_result = np.concatenate(new_result, axis=0)
            collector[oriname].append(new_result)

        merge_func = partial(_merge_func, CLASSES=self.CLASSES, iou_thr=0.1)
        if nproc <= 1:
            print('Single processing')
            merged_results = mmcv.track_iter_progress(
                (map(merge_func, collector.items()), len(collector)))
        else:
            print('Multiple processing')
            merged_results = mmcv.track_parallel_progress(
                merge_func, list(collector.items()), nproc)

        return zip(*merged_results)

