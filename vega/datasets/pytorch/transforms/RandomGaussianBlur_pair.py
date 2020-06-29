# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for RandomGaussianBlur_pair."""
import random
import cv2
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class RandomGaussianBlur_pair(object):
    """Random Gaussian Blur transform."""

    def __init__(self, kernel_size=3):
        """Construct RandomGaussianBlur_pair class.

        :param kernel_size: kernel size
        """
        self.kernel_size = kernel_size

    def __call__(self, image, label):
        """Call the transform.

        :param image: image np
        :param label: label np
        :return: transformed image and label
        """
        if random.random() < 0.5:
            image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), 0)
        return image, label
