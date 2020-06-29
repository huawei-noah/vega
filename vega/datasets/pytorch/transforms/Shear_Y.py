# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Shear_Y."""
import random
from PIL import Image
from .ops import float_parameter
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Shear_Y(object):
    """Applies ShearY to 'img'.

    The ShearY operation shears the image along the horizontal axis with 'level' magnitude.
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Shear_Y class."""
        self.level = level

    def __call__(self, img):
        """Call function of Shear_Y.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        level = float_parameter(self.level, 0.3)
        if random.random() > 0.5:
            level = -level
        return img.transform(img.size, Image.AFFINE, (1, 0, 0, level, 1, 0))
