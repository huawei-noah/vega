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

"""This is a class for Normalize_pair."""
import numpy as np
from vega.common import ClassFactory, ClassType


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
