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

"""This is a class for MaskTransform."""
import numpy as np
import mmcv
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class MaskTransform(object):
    """Mask tramsform method, which contains.

    1. resize masks to expected size and stack to a single array
    2. flip the masks (if needed)
    3. pad the masks (if needed)
    """

    def __call__(self, masks, pad_shape, scale_factor, flip=False):
        """Call function of MaskTransform.

        :param masks: mask image
        :type masks: ndarray
        :param pad_shape: (height, width)
        :type pad_shape: tuple
        :param scale_factor: the scale factor according to the image tramsform
        :type scale_factor: float
        :param flip: whether to flop or not, defaults to False
        :type flip: bool
        :return: the mask image after transform
        :rtype: ndarray
        """
        masks = [
            mmcv.imrescale(mask, scale_factor, interpolation='nearest')
            for mask in masks
        ]
        if flip:
            masks = [mask[:, ::-1] for mask in masks]
        padded_masks = [
            mmcv.impad(mask, pad_shape[:2], pad_val=0) for mask in masks
        ]
        padded_masks = np.stack(padded_masks, axis=0)
        return padded_masks
