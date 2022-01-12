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

"""This is a class for RandomGaussianBlur_pair."""
import random
import cv2
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class RandomGaussianBlur_pair(object):
    """Random Gaussian Blur transform."""

    def __init__(self, kernel_size=3):
        """Construct RandomGaussianBlur_pair class.

        :param kernel_size: kernel size
        """
        self.kernel_size = kernel_size

    def __call__(self, image, label):
        """Call the transform.

        :param image: image np
        :param label: label np
        :return: transformed image and label
        """
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return image, label
