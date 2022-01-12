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

"""Functions to implement the BiSeNet."""

import time
from collections import OrderedDict
import torch
import torch.nn as nn
from vega.modules.operators import ConvBnRelu


class AttentionRefinement(nn.Module):
    """Attention refinement module."""

    def __init__(self, in_planes, out_planes,
                 norm_layer='BN', Conv2d=nn.Conv2d):
        """Construct the AttentionRefinement class.

        :param in_planes: input channels
        :param out_planes: output channels
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        """
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   norm_layer=norm_layer,
                                   Conv2d=Conv2d)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       norm_layer=norm_layer,
                       has_relu=False, Conv2d=Conv2d),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Do an inference on AttentionRefinement.

        :param x: input tensor
        :return: output tensor
        """
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class FeatureFusion(nn.Module):
    """FeatureFusion module."""

    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=None, Conv2d=nn.Conv2d):
        """Construct the FeatureFusion class.

        :param in_planes: input channels
        :param out_planes: output channels
        :param norm_layer: type of norm layer.
        :param Conv2d: type of conv layer.
        :param reduction: reduction ratio.
        """
        super(FeatureFusion, self).__init__()
        if norm_layer is None:
            norm_layer = {'norm_type': 'BN'}
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   norm_layer=norm_layer,
                                   Conv2d=Conv2d)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       Conv2d=Conv2d),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, Conv2d=Conv2d),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        """Do an inference on FeatureFusion.

        :param x1: input tensor 1
        :param x2: input tensor 2
        :return: output tensor
        """
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


def load_model(model, model_file, is_restore=False):
    """Load model.

    :param model: created model.
    :param model_file: pretrained model file.
    :param is_restore: set true to load from model_file.
    """
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()
    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict
    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys
    if len(missing_keys) > 0:
        print('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        print('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))
    del state_dict
    t_end = time.time()
    print(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))
    return model
