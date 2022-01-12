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

"""This is a class for Rescale_pair."""
import random
import cv2
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Rescale_pair(object):
    """Rescale image and label."""

    def __init__(self, size=None, scale_range=None, scale_choices=None):
        """Construct Rescale_pair class. Must provide one of size, scale_range or scale_choices.

        :param size: size of target
        :param scale_range: range of scale
        :param scale_choices: choices of scales
        """
        if size is None and scale_range is None and scale_choices is None:
            raise ValueError("One of size, scale_range, scale_choices must be not None!")
        self.size = (size, size) if isinstance(size, int) else size
        self.scale_range = scale_range
        self.scale_choices = scale_choices

    @staticmethod
    def rescale_to_size(image, label, size):
        """Rescale image and label to a certain size.

        :param image: image np
        :param label: label np
        :param size: size as a tuple
        :return:
        """
        return (cv2.resize(image, size, interpolation=cv2.INTER_LINEAR),
                cv2.resize(label, size, interpolation=cv2.INTER_NEAREST))

    def __call__(self, image, label):
        """Call the Rescale_pair transform.

        :param image: image np
        :param label: label np
        :return: rescaled image and label
        """
        if self.size is not None:
            return self.rescale_to_size(image, label, self.size)
        else:
            if self.scale_range is not None:
                scale = random.uniform(*self.scale_range)
            else:
                scale = random.choice(self.scale_choices)
            sh = int(image.shape[0] * scale)
            sw = int(image.shape[1] * scale)
            return self.rescale_to_size(image, label, (sw, sh))
