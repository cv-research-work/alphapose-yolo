# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Jiefeng Li (jeff.lee.sjtu@gmail.com)
# -----------------------------------------------------

"""Elephant keypoint dataset."""
import os
import copy
import pdb

import numpy as np

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
import scipy.misc
import imageio

from .custom import CustomDataset
import pickle as pk


@DATASET.register_module
class Elephant(CustomDataset):
    CLASSES = ['elephant']
    EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
    14, 15,16,17,18,19]
    num_joints = 20

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joint to be swapped
        when the image is flipped horizontally."""
        return [[6,7],[8,9],[12,14],[13,15]]


    def __getitem__(self, idx):

        # get image id
        p = os.path.basename(self._items[idx])
        img_path = self._items[idx]
        img_id = img_path
        
        # load ground truth, including bbox, keypoints, image size
        label = copy.deepcopy(self._labels[idx])
        if "test" in self._ann_file:
            
            img = imageio.imread(f"data/test/{p}", pilmode='RGB')
        else:
            img = imageio.imread(img_path, pilmode='RGB')
        
 
        # transform ground truth into training label and apply data augmentation
        img, label, label_mask, bbox = self.transformation(img, label)

        return img, label, label_mask, img_id, bbox

    def _load_jsons(self):
        path = self._ann_file + '_annot_keypoint.pkl'   
        if os.path.exists(path):
            print('Lazy load annot...')
            with open(path, 'rb') as fid:
                items, labels = pk.load(fid)
        return items, labels

    


