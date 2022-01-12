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

"""This is SearchSpace for network."""
from vega.common import ClassFactory, ClassType
from vega.modules.connections import OutDictSequential
from vega.networks.necks import make_res_layer_from_code, BasicBlock
from vega.modules.operators import ops
from vega.modules.module import Module

base_arch_code = {18: '11-21-21-21',
                  34: '111-2111-211111-211',
                  50: '111-2111-211111-211',
                  101: '111-2111-21111111111111111111111-211'}


@ClassFactory.register(ClassType.NETWORK)
class ResNetDet(Module):
    """ResNet for detection."""

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (BasicBlock, (3, 4, 6, 3)),
        101: (BasicBlock, (3, 4, 23, 3))
    }

    def __init__(self, depth, num_stages=4, strides=(1, 2, 2, 2), dilations=(1, 1, 1, 1), out_indices=(0, 1, 2, 3),
                 style="pytorch", frozen_stages=-1, norm_eval=True, zero_init_residual=False, code=None):
        """Init ResNet."""
        super(ResNetDet, self).__init__()
        self.out_indices = out_indices
        self.style = style
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        if code is None:
            self.code = base_arch_code[depth].split('-')
        else:
            self.code = code.split('-')
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer_from_code(self.block, self.inplanes, planes, num_blocks,
                                                 stride=stride, dilation=dilation,
                                                 style=self.style, code=self.code[i])
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)
        self.res_layers_seq = OutDictSequential(*self.res_layers, out_list=self.out_indices)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

    def _make_stem_layer(self):
        """Make stem layer."""
        self.conv1 = ops.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = ops.BatchNorm2d(64)
        self.relu = ops.Relu(inplace=True)
        self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, padding=1)

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return self.res_layers_seq.out_channels

    def call(self, x, **kwargs):
        """Forward compute of resnet for detection."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return self.res_layers_seq(x)
