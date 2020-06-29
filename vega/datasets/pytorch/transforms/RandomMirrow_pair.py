# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is a class for RandomMirrow_pair."""
import numpy as np
from vega.core.common.class_factory import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class RandomMirrow_pair(object):
    """Random mirrow two related image."""

    def __call__(self, image, label):
        """Call function of RandomMirrow_pair.

        :param image: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type image: PIL image
        :param label: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type lebel: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        flip = np.random.choice(2) * 2 - 1
        channels_image = image.shape[-1]
        channels_label = label.shape[-1]
        if channels_image == 3:
            image = image[:, :, ::flip]
        else:
            image = image[:, ::flip]
        if channels_label == 3:
            label = label[:, :, ::flip]
        else:
            label = label[:, ::flip]
        return image, label
