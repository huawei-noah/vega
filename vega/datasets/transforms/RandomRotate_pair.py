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

"""This is a class for RandomRotate_pair."""
import random
import cv2
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class RandomRotate_pair(object):
    """Random rotate image and label."""

    def __init__(self, rotation_factor, border_value, fill_label=0):
        """Construct RandomRotate_pair class.

        :param rotation_factor: range of rotate degree
        :param border_value: value of padded border in image
        :param fill_label: value of the padded border in label
        """
        self.rotation_factor = rotation_factor
        self.border_value = border_value
        self.fill_label = fill_label

    def __call__(self, image, label):
        """Call RandomRotate_pair transform.

        :param image: image np
        :param label: label np
        :return: transformed image and label
        """
        _rotation = random.uniform(*self.rotation_factor)
        tmp_h, tmp_w = image.shape[:2]
        rotate_mat = cv2.getRotationMatrix2D((tmp_w / 2, tmp_h / 2), _rotation, 1)
        return (cv2.warpAffine(image, rotate_mat, (tmp_w, tmp_h), flags=cv2.INTER_CUBIC, borderValue=self.border_value),
                cv2.warpAffine(label, rotate_mat, (tmp_w, tmp_h), flags=cv2.INTER_NEAREST, borderValue=self.fill_label))
