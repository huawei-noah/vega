# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for SegMapTransform."""
import mmcv
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class SegMapTransform(object):
    """Semantic segmentation maps transform, which contains.

    1. rescale the segmentation map to expected size
    2. flip the image (if needed)
    3. pad the image (if needed)
    :param size_divisor:  defaults to None
    :type size_divisor: tuple
    """

    def __init__(self, size_divisor=None):
        """Construct the SegMapTransform class."""
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        """Call function of SegMapTransform."""
        if keep_ratio:
            img = mmcv.imrescale(img, scale, interpolation='nearest')
        else:
            img = mmcv.imresize(img, scale, interpolation='nearest')
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
        return img
