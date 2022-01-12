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
from vega.modules.connections import OutDictSequential, Sequential
from vega.modules.operators import ops
from vega.modules.module import Module

base_arch_code = {50: '111-2111-211111-211',
                  101: '111-2111-21111111111111111111111-211'}


class BN_Conv2d(Module):
    """Base conv2D."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = Sequential(
            ops.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                       padding=padding, dilation=dilation, groups=groups, bias=bias),
            ops.BatchNorm2d(out_channels),
            ops.Relu()
        )


class ResNeXt_Block(Module):
    """ResNeXt block with group convolutions."""

    expansion = 4

    def __init__(self, in_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls,
                               1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls,
                               3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = ops.Conv2d(
            self.group_chnls, self.group_chnls * 2, 1, stride=1, padding=0)
        self.bn = ops.BatchNorm2d(self.group_chnls * 2)
        self.short_cut = Sequential(
            ops.Conv2d(in_chnls, self.group_chnls * 2, 1, stride, 0, bias=False),
            ops.BatchNorm2d(self.group_chnls * 2)
        )

    def call(self, x):
        """Call function."""
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return ops.Relu()(out)


@ClassFactory.register(ClassType.NETWORK)
class ResNeXtDet(Module):
    """ResNet for detection."""

    arch_settings = {
        50: (ResNeXt_Block, (3, 4, 6, 3)),
        101: (ResNeXt_Block, (3, 4, 23, 3))
    }

    def __init__(self, depth, num_stages=4, strides=(1, 2, 2, 2), out_indices=(0, 1, 2, 3),
                 zero_init_residual=False, frozen_stages=-1, code=None):
        """Init ResNet."""
        super(ResNeXtDet, self).__init__()
        self.out_indices = out_indices
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        inplanes = self.inplanes
        planes = 1
        self.cardinality = 32
        if code is None:
            self.code = base_arch_code[depth].split('-')
        else:
            self.code = code.split('-')
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            res_layer, inplanes = self._make_layers(inplanes, planes, num_blocks, stride=stride, code=self.code[i])
            planes = planes * 2
            self.res_layers.append(res_layer)
        self.res_layers_seq = OutDictSequential(*self.res_layers, out_list=self.out_indices)

    def _make_layers(self, inplanes, d, blocks, stride, code):
        """Make layer."""
        strides = map(int, code)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(
                inplanes, self.cardinality, d, stride))
            inplanes = self.cardinality * d * 2
        return Sequential(*layers), inplanes

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return self.res_layers_seq.out_channels

    def _make_stem_layer(self):
        """Make stem layer."""
        self.conv1 = BN_Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def call(self, x, **kwargs):
        """Forward compute of resnet for detection."""
        x = self.conv1(x)
        x = self.maxpool(x)
        return self.res_layers_seq(x)
