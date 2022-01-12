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
