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

import logging
import math
import torch
import torch.nn as nn
from vega.common import ClassFactory, ClassType
from vega.modules.connections import OutlistSequential
from vega.networks.necks import BasicBlock, Bottleneck, ResNeXt_Block
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.modules.connections import Sequential
from vega.networks.pytorch.backbones import match_name, remove_layers, load_checkpoint
from .resnet_det import base_arch_code

base_blcok = {'BasicBlock': BasicBlock,
              'Bottleneck': Bottleneck,
              'ResNext101_32x4d': ResNeXt_Block,
              'ResNext101_64x4d': ResNeXt_Block}


@ClassFactory.register(ClassType.NETWORK)
class SpResNetDet(Module):
    """ResNet for detection."""

    def __init__(self, code=None, block='Bottleneck', parallel_code=None, subset_limit=3,
                 pretrained=None, pretrained_arch='111-2111-211111-211', depth=50):
        """Init ResNet."""
        super(SpResNetDet, self).__init__()

        self.inplanes = 64
        self._make_stem_layer()
        self.res_layers = []
        self.block = base_blcok[block]
        self.parallel_code = parallel_code
        self.subset_limit = subset_limit
        if code is None:
            self.code = base_arch_code[depth].split('-')
        else:
            self.code = code.split('-')
        self.pretrained = pretrained
        self.pretrained_arch = pretrained_arch
        self.out_indices = [i for i in range(len(self.code))]
        self.channels = [3]
        self.planes = self.inplanes
        if block == 'Bottleneck' or block == 'BasicBlock':
            self.make_resnet()
        else:
            if block == 'ResNext101_32x4d':
                self.planes = 4
                self.cardinality = 32
            elif block == 'ResNext101_64x4d':
                self.planes = 4
                self.cardinality = 64
            else:
                raise Exception("Must set correct block")
            self.make_resNext()
        self.init_weights(pretrained)

    def _make_stem_layer(self):
        """Make stem layer."""
        self.conv1 = ops.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = ops.BatchNorm2d(64)
        self.relu = ops.Relu(inplace=True)
        self.maxpool = ops.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def make_resnet(self):
        """Make resnet Net."""
        for i, code in enumerate(self.code):
            res_layer, planes = make_resnet_layer_from_code(
                self.block, self.inplanes, self.planes, code=code)
            self.channels.append(planes)
            self.inplanes = planes
            self.res_layers.append(res_layer)
            self.planes = self.planes * 2
        self.res_layers_seq = OutlistSequential(
            *self.res_layers, out_list=self.out_indices)

    def make_resNext(self):
        """Make resNext Net."""
        for i, code in enumerate(self.code):
            res_layer, planes = make_resnext_layer_from_code(self.block, self.inplanes,
                                                             self.planes, self.cardinality,
                                                             code=self.code[i])
            self.channels.append(planes)
            self.inplanes = planes
            self.res_layers.append(res_layer)
            self.planes = self.planes * 2
        self.res_layers_seq = OutlistSequential(
            *self.res_layers, out_list=self.out_indices)

    def call(self, x, **kwargs):
        """Forward compute of resnet for detection."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if self.parallel_code is None:
            outs = self.res_layers_seq(x)
        else:
            outs = []
            subset_lists = [0 for i in range(self.subset_limit)]
            parallel_numbers = self.parallel_code.split('-')
            for key, layer in enumerate(self.res_layers):
                size = x.size()[2:]
                x = layer(x)
                x_l_k = x
                for k in range(int(parallel_numbers[key])):
                    x_l_k = ops.Conv2d(
                        self.channels[key + 1], self.channels[key], kernel_size=1).cuda()(x_l_k)
                    x_l_k = ops.InterpolateScale(
                        size=size, mode='nearest')(x_l_k)
                    x_l_k = layer(subset_lists[k] + x_l_k)
                    subset_lists[k] = x_l_k
                    size = x_l_k.size()[2:]
                outs.append(x_l_k)
        return tuple(outs)

    def init_weights(self, pretrained=None):
        """Init weights."""
        if isinstance(pretrained, str):
            checkpoint = torch.load(pretrained)['weight']
            # get state_dict from checkpoint
            if 'backbone' in checkpoint:
                checkpoint = checkpoint['backbone']
            else:
                checkpoint = checkpoint

            own_state = self.state_dict()

            # when len(arch) not equal with len(pretrained arch)
            len_dis = len(self.code) - len('-'.join(self.pretrained_arch))
            if len_dis > 0:
                remove_name, bn_layers = remove_layers(
                    self.code, self.pretrained_arch, len_dis)
                for k in self.state_dict().keys():
                    for r_name in remove_name:
                        if k.find(r_name) >= 0:
                            del own_state[k]
                # turn off eval mode for all newly added bn layers
                modules = [layer[1]
                           for layer in self.named_modules() if layer[0] in bn_layers]
                for m in modules:
                    m.eval_mode = False

            pretrain_to_own = match_name(own_state.keys(), checkpoint.keys())

            mb_mapping = {}
            for mlayer in own_state.keys():
                if mlayer.find('mb') >= 0:
                    layer_n = mlayer.split('.')
                    layer_n[0] = mlayer.replace('mb', 'layer').split('_')[0]
                    layer_n = '.'.join(layer_n)
                    if layer_n in mb_mapping:
                        mb_mapping[layer_n].append(mlayer)
                    else:
                        mb_mapping[layer_n] = [mlayer]

            logger = logging.getLogger()
            load_checkpoint(self, pretrained, pretrain_to_own,
                            logger=logger, mb_mapping=mb_mapping)
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()


def make_resnet_layer_from_code(block, inplanes, planes, dilation=1,
                                with_cp=False, code=None):
    """Make resnet layer from code."""
    strides = list(map(int, code))
    layers = []
    layers.append(block(inplanes=inplanes, planes=planes, stride=strides[0], dilation=dilation,
                        with_cp=with_cp, downsample=True))
    inplanes = planes * block.expansion
    for stride in strides[1:]:
        layers.append(block(inplanes=inplanes, planes=planes, stride=stride, dilation=dilation,
                            with_cp=with_cp))
        inplanes = planes * block.expansion
    return Sequential(*layers), inplanes


def make_resnext_layer_from_code(block, inplanes, planes, cardinality=32, dilation=1,
                                 with_cp=False, code=None):
    """Make resnext layer from code."""
    strides = list(map(int, code))
    layers = []
    layers.append(block(in_chnls=inplanes, cardinality=cardinality,
                        group_depth=planes, stride=strides[0]))
    inplanes = planes * cardinality * 2
    for stride in strides[1:]:
        layers.append(block(in_chnls=inplanes, cardinality=cardinality,
                            group_depth=planes, stride=stride))
    return Sequential(*layers), inplanes
