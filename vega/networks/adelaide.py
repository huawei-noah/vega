# -*- coding:utf-8 -*-

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

"""The Adelaide model."""
from vega.networks.mobilenet import MobileNetV2Tiny
from vega.modules.blocks.micro_decoder import MicroDecoder
from vega.modules.operators import ops
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
import vega


@ClassFactory.register(ClassType.NETWORK)
class AdelaideFastNAS(Module):
    """Search space of AdelaideFastNAS."""

    def __init__(self, backbone_load_path, backbone_out_sizes, op_names, agg_size, aux_cell, sep_repeats,
                 agg_concat, num_classes, config, method, code):
        """Construct the AdelaideFastNAS class.

        :param net_desc: config of the searched structure
        """
        super(AdelaideFastNAS, self).__init__()
        self.encoder = MobileNetV2Tiny(backbone_load_path)
        self.decoder = MicroDecoder(backbone_out_sizes, op_names, num_classes, config, agg_size, aux_cell, sep_repeats,
                                    agg_concat)
        self.head = ops.InterpolateScale(mode='bilinear', align_corners=True)
        if vega.is_ms_backend():
            self.permute = ops.Permute((0, 2, 3, 1))

    def call(self, inputs):
        """Do an inference on AdelaideFastNAS model."""
        self.head.size = ops.get_shape(inputs)[2:]
        return super().call(inputs)
