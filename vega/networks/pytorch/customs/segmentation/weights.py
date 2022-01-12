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

"""Weight operations of the norm layers."""
from torch import nn


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    """Init weight of layers.

    :param feature: layers of model
    :param conv_init: conv init function
    :param norm_layer: type of norm layer.
    :param bn_eps: eps of bn layer.
    :param bn_momentum: momentum of bn layer.
    """
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            if hasattr(m, 'momentum') and bn_momentum is not None:
                m.momentum = bn_momentum


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    """Init weight of model.

    :param module_list: module list of model
    :param conv_init: conv init function
    :param norm_layer: type of norm layer.
    :param bn_eps: eps of bn layer.
    :param bn_momentum: momentum of bn layer.
    """
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)
