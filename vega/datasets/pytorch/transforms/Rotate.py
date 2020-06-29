# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Rotate."""
import random
from .ops import int_parameter
from vega.core.common.class_factory import ClassFactory, ClassType


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
