# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Param class."""
import re


class ParamsFactory:
    """Class of param.

    :param detections_type: list of strings - Selected types on which the evaluation is
                            performed (i.e: detector of this type is evaluated)
    :param difficulty: the selected difficulty to be evaluated
    :param ignore_other_vru: - If true, other VRU ground truth boxes are used during the
                               matching process, therefore other classes (e.g. riders)
                               which are detected and classified as the primary detection class
                               (e.g. pedestrians) do not cause a false positive.
                             - if true other VRUs (see tolerated_other_classes) are marked with
                               the ignored flag, otherwise they are discarded
    :param tolerated_other_classes: list of strings - Other classes which are tolerated,
                                    if the ignore_other_vru flag is set.
    :param dont_care_classes: list of strings - don't care region class name
    """

    def __init__(self,
                 detections_type,
                 difficulty,
                 ignore_other_vru,
                 tolerated_other_classes=[],
                 dont_care_classes=[],
                 ignore_type_for_skipped_gts=1,
                 size_limits={'reasonable': 40, 'small': 30, 'occluded': 40, 'all': 20},
                 occ_limits={'reasonable': 40, 'small': 40, 'occluded': 80, 'all': 80},
                 size_upper_limits={'small': 60},
                 occ_lower_limits={'occluded': 40},
                 rider_boxes_including_vehicles=False,
                 discard_depictions=False,
                 clipping_boxes=False,
                 transform_det_to_xy_coordinates=False
                 ):
        self.difficulty = difficulty
        self.ignore_other_vru = ignore_other_vru
        self.tolerated_other_classes = tolerated_other_classes
        self.dont_care_classes = dont_care_classes
        self.ignore_type_for_skipped_gts = ignore_type_for_skipped_gts
        self.detections_type = detections_type
        self.size_limits = size_limits
        self.occ_limits = occ_limits
        self.size_upper_limits = size_upper_limits
        self.occ_lower_limits = occ_lower_limits
        self.rider_boxes_including_vehicles = rider_boxes_including_vehicles
        self.discard_depictions = discard_depictions
        self.clipping_boxes = clipping_boxes
        self.transform_det_to_xy_coordinates = transform_det_to_xy_coordinates

    def ignore_gt(self, gt):
        """Ignore gt."""
        h = gt['y1'] - gt['y0']

        if gt['identity'] in self.detections_type:
            pass
        elif self.ignore_other_vru and gt['identity'] in self.tolerated_other_classes:
            return 1
        elif gt['identity'] in self.dont_care_classes:
            if self.discard_depictions and gt['identity'] == 'person-group-far-away' and \
                    'depiction' in gt['tags']:
                return None
            else:
                return 2
        else:
            return None
        if gt['identity'] == 'pedestrian':
            for tag in gt['tags']:
                if tag in ['sitting-lying', 'behind-glass']:
                    return 1
        truncation = 0
        occlusion = 0
        for t in gt['tags']:
            if 'occluded' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    occlusion = int(matches[0])
            elif 'truncated' in t:
                matches = re.findall(r'\d+', t)
                if len(matches) == 1:
                    truncation = int(matches[0])
        return self.judge_ignored_gt(h, truncation, occlusion)

    def judge_ignored_gt(self, h, truncation, occlusion):
        """Judge type for skipped gts."""
        if h < self.size_limits[self.difficulty] or \
                occlusion >= self.occ_limits[self.difficulty] or \
                truncation >= self.occ_limits[self.difficulty]:

            return self.ignore_type_for_skipped_gts

        if self.difficulty in self.size_upper_limits:
            if h > self.size_upper_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts

        if self.difficulty in self.occ_lower_limits:
            if occlusion < self.occ_lower_limits[self.difficulty]:
                return self.ignore_type_for_skipped_gts

        return 0

    def preprocess_gt(self, gt):
        """Process gt."""
        if str(gt['identity']).lower() == 'rider' and self.rider_boxes_including_vehicles:
            for subent in gt['children']:
                gt['x0'] = min(gt['x0'], float(subent['x0']))
                gt['y0'] = min(gt['y0'], float(subent['y0']))
                gt['x1'] = max(gt['x1'], float(subent['x1']))
                gt['y1'] = max(gt['y1'], float(subent['y1']))
        if self.clipping_boxes:
            gt['y0'] = max(0, gt['y0'])
            gt['y1'] = min(1024, gt['y1'])
            gt['x0'] = max(0, gt['x0'])
            gt['x1'] = min(1920, gt['x1'])

    def skip_gt(self, gt):
        """Skip gt."""
        if self.ignore_gt(gt) is None:
            return True
        return False

    def preprocess_det(self, det):
        """Process det."""
        assert self.transform_det_to_xy_coordinates
        if self.transform_det_to_xy_coordinates and 'maxrow' in det:
            det['x0'] = det['mincol']
            det['y0'] = det['minrow']
            det['x1'] = det['maxcol'] + 1
            det['y1'] = det['maxrow'] + 1
            det.pop('maxrow')
            det.pop('minrow')
            det.pop('maxcol')
            det.pop('mincol')
            assert 'maxrow' not in det
        if self.clipping_boxes:
            det['y0'] = max(0, det['y0'])
            det['y1'] = min(1024, det['y1'])
            det['x0'] = max(0, det['x0'])
            det['x1'] = min(1920, det['x1'])
        if det['identity'] == 'cyclist':
            det['identity'] = 'rider'

    def skip_det(self, det):
        """Skip det."""
        assert 'y1' in det
        wrong_type = det['identity'] not in self.detections_type
        expFilter = 1.25
        height = det['y1'] - det['y0']
        too_small = height <= self.size_limits[self.difficulty] / expFilter
        too_high = False if self.difficulty not in self.size_upper_limits else \
            height >= self.size_upper_limits[self.difficulty] * expFilter
        return wrong_type or too_small or too_high
