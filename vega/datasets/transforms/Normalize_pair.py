# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Normalize_pair."""
import numpy as np
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class Normalize_pair(object):
    """Normalize image in (image, label) pair."""

    def __init__(self, mean, std):
        """Construct Normalize_pair class.

        :param mean: bgr mean
        :param std: bgr std
        """
        self.mean = mean
        self.std = std

    def __call__(self, image, label):
        """Call Normalize_pair transform.

        :param image: image as np.array
        :param label: label as np.array
        :return: normalized image and label
        """
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return image, label
