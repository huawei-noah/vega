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

"""This is a class for ToPILImage_pair."""
from torchvision.transforms import functional as F
from vega.common import ClassFactory, ClassType


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
