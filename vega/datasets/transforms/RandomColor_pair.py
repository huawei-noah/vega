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
import cv2
import numpy as np
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class RandomColor_pair(object):
    """Random crop image and label."""

    def __init__(self, color_factor, contrast_factor, brightness_factor):
        """Construct RandomColor_pair class.

        :param color_factor: color_factor range
        :param contrast_factor: contrast_factor range
        :param brightness_factor: brightness_factor range
        """
        self.color_factor = color_factor
        self.contrast_factor = contrast_factor
        self.brightness_factor = brightness_factor

    def __call__(self, image, label):
        """Call the transform.

        :param image: image np
        :param label: label np
        :return: transformed image and label
        """
        _color = random.uniform(*self.color_factor)
        _contrast = random.uniform(*self.contrast_factor)
        _brightness = random.uniform(*self.brightness_factor)
        _HSV = np.dot(cv2.cvtColor(image, cv2.COLOR_BGR2HSV).reshape((-1, 3)),
                      np.array([[_color, 0, 0], [0, _contrast, 0], [0, 0, _brightness]]))
        _HSV_H = np.where(_HSV < 255, _HSV, 255)
        image = cv2.cvtColor(np.uint8(_HSV_H.reshape((-1, image.shape[1], 3))), cv2.COLOR_HSV2BGR)
        return image, label
