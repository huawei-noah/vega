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

"""This is a class for Posterize."""
from PIL import ImageOps
from vega.common import ClassFactory, ClassType
from .ops import int_parameter


@ClassFactory.register(ClassType.TRANSFORM)
class Posterize(object):
    """Applies Posterize to 'img'.

    The Posterize operation reduces the number of bits for each pixel with 'level' magnitude.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Posterize class."""
        self.level = level

    def __call__(self, img):
        """Call function of Posterize.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        level = int_parameter(self.level, 4)
        return ImageOps.posterize(img.convert('RGB'), 4 - level)
