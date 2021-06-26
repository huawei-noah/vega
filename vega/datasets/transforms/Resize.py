# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Color."""
from PIL import Image
from vega.common import ClassFactory, ClassType
from collections import Iterable


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
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size[::-1], self.interpolation)
