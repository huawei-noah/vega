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

"""This is a class for RandomCrop_pair."""
import random
import numpy as np
import cv2
from vega.common import ClassFactory, ClassType
from PIL import Image


@ClassFactory.register(ClassType.TRANSFORM)
class RandomCrop(object):
    """Random crop method for images.

    :param crop: crop size
    :type crop: np.array
    """

    def __init__(self, size, padding=0, pad_img=0):
        """Construct RandomCrop_pair class."""
        self.crop = [size, size] if isinstance(size, int) else size
        self.crop = np.array(self.crop)
        self.padding = padding
        self.pad_img = pad_img

    def __call__(self, image):
        """Call RandomCrop_pair transform to random crop two images.

        :param image: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type image: PIL image
        :param label: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type lebel: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        h, w = image.size
        start_h = random.randint(0, h - self.crop[0]) if h > self.crop[0] else 0
        start_w = random.randint(0, w - self.crop[1]) if w > self.crop[1] else 0
        borders = np.array([start_h, start_w, self.crop[0] + start_h, self.crop[1] + start_w])
        image = np.array(image)
        image = image[borders[0]:borders[2], borders[1]:borders[3]]
        image = self._pad_image_to_shape(image, self.crop, self.pad_img)
        return Image.fromarray(image)

    @staticmethod
    def _pad_image_to_shape(data, shape, value):
        """Pad value to the data.

        :param data: data np
        :param shape: shape of target
        :param value: pad value
        :return: padded data
        """
        pad_height = shape[0] - data.shape[0]
        pad_width = shape[1] - data.shape[1]
        if pad_height > 0 or pad_width > 0:
            pad_top, pad_left = pad_height // 2, pad_width // 2
            pad_bottom, pad_right = pad_height - pad_top, pad_width - pad_left
            return cv2.copyMakeBorder(data, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=value)
        else:
            return data
