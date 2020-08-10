# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ECP DATA API."""

import glob
import json
import os
import re
import numpy as np
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.registry import DATASETS
from pycocotools.coco import COCO


@DATASETS.register_module
class ECPDataset(CocoDataset):
    """ECPDataset."""

    CLASSES = ("pedestrian", "rider")

    def load_annotations(self, ann_file):
        """Load and initialize annotations."""
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.CLASSES)
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        filter_imgs = []

        # filter some proposals
        ids_with_ann = set()
        for _ in self.coco.anns.values():
            # filter objects with others classes and area < 20
            if _['ignore'] is None or _['area'] < 20:
                continue
            # elif _['ignore'] >= 0: # only consider sample with 'ignore' == 0
            ids_with_ann.add(_['image_id'])

        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                filter_imgs.append(self.img_ids[i])
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        print("filter {} images".format(len(filter_imgs)))
        return valid_inds

    def get_ann_info(self, idx):
        """Get annotation."""
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.cat_ids)
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    @classmethod
    def load_gt_det(cls, gt_path, det_path, gt_ext='.json', det_ext='.json'):
        """Load gt and det.

        :param gt_path: path of ground truth
        :type gt_path: str
        :param det_path: path of det
        :type det_path: str
        :param gt_ext: type of gt file
        :type gt_ext: str
        :param det_ext: type of det file
        :type det_ext: str
        :return: data of gt and det
        :rtype: dict
        """
        if not os.path.isdir(det_path):
            raise IOError('{} is not a directory.'.format(det_path))
        if gt_path.endswith('.dataset'):
            with open(gt_path) as datasetf:
                gt_files = datasetf.readlines()
                gt_files = [f.strip() for f in gt_files if len(f) > 0]
        else:
            gt_files = glob.glob(gt_path + '/*' + gt_ext)
            if len(gt_files) == 0:
                gt_files = glob.glob(gt_path + '/*/*' + gt_ext)
        gt_files.sort()
        if not gt_files:
            raise ValueError('ERROR: No ground truth files found at given location! ABORT.'
                             'Given path was: {} and gt ext looked for was {}'.format(gt_path, gt_ext))
        det_files = glob.glob(det_path + '/*' + det_ext)
        if not det_files:
            det_files = glob.glob(det_path + '/*/*' + det_ext)
            if not det_files:
                det_files = glob.glob(det_path + '/*/*/*' + det_ext)
        det_files.sort()
        if not det_files:
            raise ValueError('ERROR: No ground truth files found at given location! ABORT.'
                             'Given path was: {} and gt ext looked for was {}'.format(det_path,
                                                                                      det_ext))
        if len(gt_files) != len(det_files):
            raise ValueError('Number of detection json files {} does not match the number of '
                             'ground truth json files {}.\n'
                             'Please provide for each image in the ground truth set one detection file.'
                             .format(len(det_files), len(gt_files)))
        for gt_file, det_file in zip(gt_files, det_files):
            gt_fn = os.path.basename(gt_file)
            det_fn = os.path.basename(det_file)

            gt_frame_id = re.search('(.*?)' + gt_ext, gt_fn).group(1)
            det_frame_id = re.search('(.*?)' + det_ext, det_fn).group(1)
            if gt_frame_id != det_frame_id:
                raise ValueError('Error: Frame identifiers do not match: "{}" vs. "{}".'
                                 'Check number and order of files in'
                                 ' ground truth and detection folder. ABORT.'.format(gt_frame_id,
                                                                                     det_frame_id))
            with open(gt_file, 'rb') as f:
                gt = json.load(f)
            gt_frame = cls.get_gt_frame(gt)
            for gt in gt_frame['children']:
                cls._prepare_ecp_gt(gt)
            with open(det_file, 'rb') as f:
                det = json.load(f)
            det_frame = cls.get_det_frame(det)
            det_frame['filename'] = det_file
            for det in det_frame['children']:
                cls._prepare_det(det)

            yield {'gt': gt_frame, 'det': det_frame}

    @classmethod
    def get_gt_frame(cls, gt_dict):
        """Get frame of gt."""
        if gt_dict['identity'] == 'frame':
            pass
        elif '@converter' in gt_dict:
            gt_dict = gt_dict['children'][0]['children'][0]
        elif gt_dict['identity'] == 'seqlist':
            gt_dict = gt_dict['children']['children']

        assert gt_dict['identity'] == 'frame'
        return gt_dict

    @classmethod
    def get_det_frame(cls, det_dict):
        """Get frame of det."""
        if '@converter' in det_dict:
            det_dict = det_dict['children'][0]['children'][0]
        elif 'objects' in det_dict:
            det_dict['children'] = det_dict['objects']
        return det_dict

    @classmethod
    def _prepare_ecp_gt(cls, gt):
        """Prepare gt of ecp.

        :param gt: ground truth
        :type gt: dict
        """
        def translate_ecp_pose_to_image_coordinates(angle):
            angle = angle + 90.0
            angle = angle % 360
            if angle > 180:
                angle -= 360.0
            return np.deg2rad(angle)
        orient = None
        if gt['identity'] == 'rider':
            if len(gt['children']) > 0:  # vehicle is annotated
                for cgt in gt['children']:
                    if cgt['identity'] in ['bicycle', 'buggy', 'motorbike', 'scooter', 'tricycle',
                                           'wheelchair']:
                        orient = cgt.get('Orient', None) or cgt.get('orient', None)
        else:
            orient = gt.get('Orient', None) or gt.get('orient', None)
        if orient:
            gt['orient'] = translate_ecp_pose_to_image_coordinates(orient)
            gt.pop('Orient', None)

    @classmethod
    def _prepare_det(cls, det):
        """Prepare det.

        :param det: det of inference
        :type: dict
        """
        if 'score' not in det.keys():
            score = det.get('confidencevalues', [None])[0]
            if score is None:
                raise ValueError('Missing key "score" in detection {}'.format(det))
            det['score'] = score
        orient = det.get('orient', None) or det.get('Orient', None)
        if orient:
            det['orient'] = orient
            det.pop('Orient', None)
