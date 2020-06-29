# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Sharpness."""
from PIL import ImageEnhance
from .ops import float_parameter
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Sharpness(object):
    """Applies Sharpness to 'img'.

    The Sharpness operation adjusts the sharpness of the image, level = 0 gives a blurred image,
    whereas level = 1 gives the original image
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, level):
        """Construct the Sharpness class."""
        self.level = level

    def __call__(self, img):
        """Call function of Sharpness.

        :param img: input image
        :type img: numpy or tensor
        :return: the image after transform
        :rtype: numpy or tensor
        """
        v = float_parameter(self.level, 1.8) + .1
        return ImageEnhance.Sharpness(img).enhance(v)
