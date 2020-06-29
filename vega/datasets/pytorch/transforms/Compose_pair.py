# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for Compose_pair."""


class Compose_pair(object):
    """Composes several transforms together.

    :param transforms: transform method
    :type transforms: callable class
    """

    def __init__(self, transforms):
        """Construct the Compose_pair class."""
        self.transforms = transforms

    def __call__(self, img1, img2):
        """Call function of Compose_pair.

        :param image: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type image: PIL image
        :param label: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type lebel: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        for t in self.transforms:
            img1, img2 = t(img1, img2)
        return img1, img2
