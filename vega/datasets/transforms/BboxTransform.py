# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is a class for BboxTransform."""
import numpy as np
from vega.common import ClassFactory, ClassType


def bbox_flip(bboxes, img_shape):
    """Flip bboxes horizontally.

    :param bboxes: shape (..., 4*k)
    :type bboxes: ndarray
    :param img_shape:(height, width)
    :type img_shape: tuple
    :return: the bbox after flip
    :rtype: ndarray
    """
    w = img_shape[1]
    flipped = bboxes.copy()
    flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
    flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
    return flipped


@ClassFactory.register(ClassType.TRANSFORM)
class BboxTransform(object):
    """Bbox transform, which contains.

    1. rescale bboxes according to image size
    2. flip bboxes (if needed)
    3. pad the first dimension to `max_num_gts`
    :param max_num_gts: the maxmum number of the ground truth, defaults to None
    :type max_num_gts: int
    """

    def __init__(self, max_num_gts=None):
        """Construct the BboxTransform class."""
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, scale_factor, flip=False):
        """Call function of BboxTransform.

        :param bboxes: bounding box
        :type bboxes: ndarray
        :param img_shape: image shape
        :type img_shape: tuple
        :param scale_factor: the scale factor according to the image tramsform
        :type scale_factor: float
        :param flip: Whether to flip or not, defaults to False
        :type flip: bool
        :return: the bounding box after tramsform
        :rtype: ndarray
        """
        gt_bboxes = bboxes * scale_factor
        if flip:
            gt_bboxes = bbox_flip(gt_bboxes, img_shape)
        gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0, img_shape[1] - 1)
        gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0, img_shape[0] - 1)
        if self.max_num_gts is None:
            return gt_bboxes
        else:
            num_gts = gt_bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 4), dtype=np.float32)
            padded_bboxes[:num_gts, :] = gt_bboxes
            return padded_bboxes
