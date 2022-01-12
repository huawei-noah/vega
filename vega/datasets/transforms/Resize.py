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

"""This is a class for Color."""
from collections import Iterable
from PIL import Image
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Resize(object):
    """Applies Color to 'img'.

    The Color operation adjusts the color balance of the image, level = 0 gives a black & white image,
    whereas level = 1 gives the original image
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        """Construct the Color class."""
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """Call function of Resize.

        :param img: input image
        :type img: PIL Image
        :return: the image after transform
        :rtype: numpy or tensor
        """
        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
        if not (isinstance(self.size, int) or (isinstance(self.size, Iterable) and len(self.size) == 2)):
            raise TypeError('Got inappropriate size arg: {}'.format(self.size))

        if isinstance(self.size, int):
            width, height = img.size
            if (height <= width and height == self.size) or (width <= height and width == self.size):
                return img
            if width > height:
                oh = self.size
                ow = int(self.size * width / height)
                return img.resize((ow, oh), self.interpolation)
            else:
                ow = self.size
                oh = int(self.size * height / width)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)
