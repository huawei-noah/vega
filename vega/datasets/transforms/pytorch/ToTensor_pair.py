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

"""This is a class for ToTensor_pair."""
import torch
from torchvision.transforms import functional as F
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.TRANSFORM)
class ToTensor_pair(object):
    """Tranform two PIL image to tensor."""

    def __call__(self, img1, img2):
        """Call function of ToTensor_pair.

        :param img1: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type img1: PIL image
        :param img2: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type img2: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        return F.to_tensor(img1), F.to_tensor(img2)


@ClassFactory.register(ClassType.TRANSFORM)
class ToTensorAll(object):
    """Transform all inputs to tensor."""

    def __call__(self, *inputs):
        """Call function of ToTensorAll."""
        return tuple([torch.tensor(data, dtype=torch.long) for data in inputs])


@ClassFactory.register(ClassType.TRANSFORM)
class PILToTensorAll(object):
    """Transform PIL image to tensor."""

    def __call__(self, *imgs):
        """Call function of ToTensor_pair.

        :param img1: usually the feature image, for example, the LR image for super solution dataset,
        the initial image for the segmentation dataset, and etc
        :type img1: PIL image
        :param img2: usually the label image, for example, the HR image for super solution dataset,
        the mask image for the segmentation dataset, and etc
        :type img2: PIL image
        :return: the image after transform
        :rtype: list, erery item is a PIL image, the first one is feature image, the second is label image
        """
        res = [img if isinstance(img, dict) or isinstance(img, list) else F.to_tensor(img) for img in imgs]
        return tuple(res) if len(res) > 1 else res
