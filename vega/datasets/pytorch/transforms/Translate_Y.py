# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Translate_Y."""
import random
from PIL import Image
from .ops import int_parameter
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Translate_Y(object):
    """Applies TranslateY to 'img'.

    The TranslateY operation translates the image in the vertical direction by 'level' number of pixels.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Translate_Y class."""
        self.level = level

    def __call__(self, img):
        """Call function of Translate_Y.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        level = int_parameter(self.level, 10)
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, level))
