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
import numpy as np
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Normalize(object):
    """Applies Color to 'img'.

    The Color operation adjusts the color balance of the image, level = 0 gives a black & white image,
    whereas level = 1 gives the original image
    :param level: Strength of the operation specified as an Integer from [0, 'PARAMETER_MAX'].
    :type level: int
    """

    def __init__(self, mean, std):
        """Construct the Color class."""
        self.mean = np.array(mean).reshape(3, 1, 1)
        self.std = np.array(std).reshape(3, 1, 1)

    def __call__(self, img):
        """Call function of Normalize.

        :param img: input image
        :type img: PIL Image
        :return: the image after transform
        :rtype: numpy or tensor
        """
        if isinstance(img, Image.Image):
            img = np.array(img)
        data_type = img.dtype
        self.mean = self.mean.astype(data_type)
        self.std = self.std.astype(data_type)
        img = (img - self.mean) / self.std
        return img
