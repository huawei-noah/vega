# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
