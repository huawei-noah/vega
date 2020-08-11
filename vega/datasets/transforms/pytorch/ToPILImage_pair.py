# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for ToPILImage_pair."""
from torchvision.transforms import functional as F
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class ToPILImage_pair(object):
    """Tranform two tenor to PIL image."""

    def __init__(self, mode=None):
        """Construct the ToPILImage_pair class."""
        self.mode = mode

    def __call__(self, img1, img2):
        """Call function of ToPILImage_pair.

        :param img1: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type img1: PIL image
        :param img2: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type img2: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        return F.to_pil_image(img1, self.mode), F.to_pil_image(img2, self.mode)
