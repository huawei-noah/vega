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

"""This is a class for Rotate."""
import random
from vega.common import ClassFactory, ClassType
from .ops import int_parameter


@ClassFactory.register(ClassType.TRANSFORM)
class Rotate(object):
    """Applies Rotates to 'img'.

    The Rotate operation rotates the image from -30 to 30 degrees depending on 'level'.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Rotate class."""
        self.level = level

    def __call__(self, img):
        """Call function of Rotate.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        degrees = int_parameter(self.level, 30)
        if random.random() > 0.5:
            degrees = -degrees
        return img.rotate(degrees)
