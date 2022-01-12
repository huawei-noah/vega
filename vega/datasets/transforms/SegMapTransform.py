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

"""This is a class for SegMapTransform."""
import mmcv
from vega.common import ClassFactory, ClassType


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
