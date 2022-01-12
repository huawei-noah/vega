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

"""Defined faster rcnn detector."""

from object_detection.anchor_generators import grid_anchor_generator
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class AnchorGenerator(object):
    """Anchor Generator."""

    def __init__(self, desc):
        """Init AnchorGenerator.

        :param desc: config dict
        """
        self.model = None
        self.height = 256  # Default
        self.width = 256  # Default
        self.height_offset = 0  # Default
        self.width_offset = 0  # Default
        self.type = desc.type
        self.scales = desc.scales
        self.aspect_ratios = desc.aspect_ratios
        self.height_stride = desc.height_stride
        self.width_stride = desc.width_stride

    def get_real_model(self, training):
        """Get real model of AnchorGenerator."""
        if self.model:
            return self.model
        else:
            self.model = grid_anchor_generator.GridAnchorGenerator(
                scales=[float(scale) for scale in self.scales],
                aspect_ratios=[float(aspect_ratio)
                               for aspect_ratio
                               in self.aspect_ratios],
                base_anchor_size=[self.height, self.width],
                anchor_stride=[self.height_stride, self.width_stride],
                anchor_offset=[self.height_offset, self.width_offset])
            return self.model

    def __call__(self, features, labels, training):
        """Forwad function of AnchorGenerator."""
        return self.get_real_model(training).predict(features, labels)
