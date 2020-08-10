# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Get IOU."""


class GetIou:
    """Class of IOU.

    :param gt: ground truth
    :type gt: dict
    :param det: det
    :type det: dict
    """

    def __init__(self, gt, det, image_height=0, image_width=0):

        self.gt = gt
        self.det = det
        self.gt_max_x = max(gt['x1'], gt['x0'])
        self.gt_max_y = max(gt['y1'], gt['y0'])
        self.gt_min_x = min(gt['x1'], gt['x0'])
        self.gt_min_y = min(gt['y1'], gt['y0'])

        self.det_max_x = det['x1']
        self.det_max_y = det['y1']
        self.det_min_x = det['x0']
        self.det_min_y = det['y0']

        assert self.det_max_x >= self.det_min_x, 'Invalid detection: mincol > maxcol'
        assert self.det_max_y >= self.det_min_y, 'Invalid detection: minrow > maxrow'
        assert self.gt_max_x >= self.gt_min_x, 'Invalid ground truth: mincol > maxcol'
        assert self.gt_max_y >= self.gt_min_y, 'Invalid ground truth: minrow > maxrow'

        self.gt_ignore = gt.get('ignore', False)

        self.image_height = image_height
        self.image_width = image_width
        self.img_size = self.image_height * self.image_width

        self._tp = 0
        self._fp = 0
        self._fn = 0

        self._calculated = {
            'tp': False,
            'fp': False,
            'fn': False,
        }
        self.match = False

        self.get_iou()

    @property
    def tp_pixels(self):
        """Get tp."""
        if not self._calculated['tp']:
            w_inter = min(self.gt_max_x, self.det_max_x) - max(self.gt_min_x, self.det_min_x)
            h_inter = min(self.gt_max_y, self.det_max_y) - max(self.gt_min_y, self.det_min_y)
            if w_inter <= 0 or h_inter <= 0:
                self._tp = 0
            else:
                self._tp = w_inter * h_inter
            self._calculated['tp'] = True
        return self._tp

    @property
    def fp_pixels(self):
        """Get fp."""
        if not self._calculated['fp']:
            det_h = self.det_max_y - self.det_min_y
            det_w = self.det_max_x - self.det_min_x
            self._fp = det_h * det_w - self.tp_pixels
            self._calculated['fp'] = True
        return self._fp

    @property
    def fn_pixels(self):
        """Get fn."""
        if not self._calculated['fn']:
            gt_h = self.gt_max_y - self.gt_min_y
            gt_w = self.gt_max_x - self.gt_min_x
            self._fn = gt_h * gt_w - self.tp_pixels
            self._calculated['fn'] = True
        return self._fn

    def get_iou(self):
        """Calculate iou."""
        if self.tp_pixels == 0:
            return 0.0
        if self.gt['ignore'] == 2:
            all_pixels = self.fp_pixels + self.tp_pixels
        elif self.gt['ignore'] == 3:
            all_pixels = self.fn_pixels + self.tp_pixels
        else:
            all_pixels = self.tp_pixels + self.fp_pixels + self.fn_pixels
        self.iou = self.tp_pixels / float(all_pixels)
        if self.iou >= 0.5:
            self.match = True

    def better_match(self, before_iou):
        """Compare iou."""
        if before_iou is None:
            return True
        elif self.iou >= before_iou:
            return True
        else:
            return False
