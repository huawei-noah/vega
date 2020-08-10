# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNet models for detection."""
import torch.nn as nn
import torch.nn.functional as F
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory
from ..blocks.conv_module import ConvModule


@NetworkFactory.register(NetTypes.NECK)
class FPN(Network):
    """FPN."""

    def __init__(self, desc):
        """Init FPN.

        :param desc: config dict
        """
        super(FPN, self).__init__()
        self.in_channels = desc["in_channels"]
        self.out_channels = desc["out_channels"]
        self.num_ins = len(self.in_channels)
        self.num_outs = desc["num_outs"]
        self.activation = desc["activation"] if "activation" in desc else None
        self.start_level = desc["start_level"] if "start_level" in desc else 0
        self.end_level = desc["end_level"] if "end_level" in desc else -1
        self.add_extra_convs = desc["add_extra_convs"] if "add_extra_convs" in desc else None
        self.extra_convs_on_inputs = desc["extra_convs_on_inputs"] if "extra_convs_on_inputs" in desc else None
        self.relu_before_extra_convs = desc["relu_before_extra_convs"] if "relu_before_extra_convs" in desc else None
        self.conv_cfg = desc["conv_cfg"] if "conv_cfg" in desc else {"type": "Conv"}
        self.norm_cfg = desc["norm_cfg"] if "norm_cfg" in desc else {"type": "BN", "requires_grad": True}
        if self.end_level == -1:
            self.backbone_end_level = self.num_ins
            assert self.num_outs >= self.num_ins - self.start_level
        else:
            self.backbone_end_level = self.end_level
            assert self.end_level <= len(self.in_channels)
            assert self.num_outs == self.end_level - self.start_level
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                self.in_channels[i],
                self.out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False)
            fpn_conv = ConvModule(
                self.out_channels,
                self.out_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                activation=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = self.out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    self.out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Init weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """Forward compute.

        :param inputs: input feature map
        :return: tuple of feature map
        """
        # assert len(inputs) == len(self.in_channels)
        laterals = [lateral_conv(inputs[i + self.start_level]) for i, lateral_conv in enumerate(self.lateral_convs)]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
