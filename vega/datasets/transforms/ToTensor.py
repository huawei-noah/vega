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
