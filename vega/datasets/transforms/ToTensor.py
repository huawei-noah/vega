# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for RandomCrop_pair."""
from PIL import Image
import numpy as np
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class ToTensor(object):
    """Random crop method for images.

    :param crop: crop size
    :type crop: np.array
    """

    def __call__(self, img):
        """Call RandomCrop_pair transform to random crop two images.

        :param image: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if img.mode not in ["I", "I;16", "F", "1"]:
            img = np.array(img).astype(np.float32) / 255
        else:
            img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        return img
