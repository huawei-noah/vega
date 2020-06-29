# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SpFPN Module."""

from mmdet.models.necks import FPN
from mmdet.models.registry import NECKS
import torch.nn.functional as F
from mmdet.core import auto_fp16


@NECKS.register_module
class SpFPN(FPN):
    """Class of FPN_.

    :param in_channels: input feature map channel num
    :type in_channels: int

    :param out_channels: output feature map channel num
    :type out_channels: int

    :param num_outs: num output
    :type num_outs: int

    :param start_level: start level
    :type start_level: int

    :param end_level: end level
    :type end_level: int

    :param conv_cfg: conv config
    :type conv_cfg: dict

    :param norm_cfg: norm config
    :type norm_cfg: dict

    :param activation: activation
    :type activation: dict
    """

    def __init__(self, in_channels, out_channels, num_outs):
        super(SpFPN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            conv_cfg=None,
            norm_cfg=None,
            activation=None)

    @auto_fp16()
    def forward(self, inputs):
        """Forward compute.

        :param inputs: input feature
        :type inputs: torch.Tensor
        :return: output feature map
        :rtype: torch.Tensor
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            try:
                laterals[i - 1] += F.interpolate(
                    laterals[i], scale_factor=2, mode='nearest')
            except:
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=laterals[i - 1].size()[2:], mode='nearest')

        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        if self.num_outs > len(outs):
            for i in range(self.num_outs - used_backbone_levels):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
