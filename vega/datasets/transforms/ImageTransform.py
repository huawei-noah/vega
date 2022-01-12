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

"""This is a class for ImageTransform."""
import numpy as np
import mmcv
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class ImageTransform(object):
    """Image transform method, which contains.

    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    :param mean: the mean value to normalized , defaults to (0, 0, 0)
    :type mean: tuple, optional
    :param std: the std value to normalized, defaults to (1, 1, 1)
    :type std: tuple, optional
    :param to_rgb: whether the mode of the image is rgb or not, defaults to True
    :type to_rgb: bool, optional
    :param size_divisor: pad shape, defaults to None
    :type size_divisor: int, optional
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 to_rgb=True,
                 size_divisor=None):

        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size_divisor = size_divisor

    def __call__(self, img, scale, flip=False, keep_ratio=True):
        """Call function of ImageTransform.

        :param img: input image
        :type img: numpy or tensor
        :param scale: a random scaler
        :type scale: float
        :param flip: wheather flip or not, defaults to False
        :type flip: bool, optional
        :param keep_ratio: whether to keep the aspect ratio or not, defaults to True
        :type keep_ratio: bool, optional
        :return: the image after transform and other paras
        :rtype: list
        """
        if keep_ratio:
            img, scale_factor = mmcv.imrescale(img, scale, return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img, scale, return_scale=True)
            scale_factor = np.array(
                [w_scale, h_scale, w_scale, h_scale], dtype=np.float32)
        img_shape = img.shape
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        if flip:
            img = mmcv.imflip(img)
        if self.size_divisor is not None:
            img = mmcv.impad_to_multiple(img, self.size_divisor)
            pad_shape = img.shape
        else:
            pad_shape = img_shape
        img = img.transpose(2, 0, 1)
        return img, img_shape, pad_shape, scale_factor
