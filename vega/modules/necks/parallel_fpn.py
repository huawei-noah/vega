# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""This is Network for SerialNet."""
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.modules.connections import ModuleList


@ClassFactory.register(ClassType.NETWORK)
class ParallelFPN(Module):
    """Parallel FPN."""

    def __init__(self, in_channels=[64, 128, 256, 512], out_channels=256, code=None,
                 weight_file=None, weights_prefix='head.backbone.1'):
        """Init FPN.

        :param desc: config dict
        """
        super(ParallelFPN, self).__init__()
        self.code = code
        self.lateral_convs = ModuleList()
        self.fpn_convs = ModuleList()
        self.weight_file = weight_file
        self.weights_prefix = weights_prefix
        for in_channel in in_channels:
            # l_conv = Sequential(ops.Conv2d(in_channel, out_channels, 1), ops.BatchNorm2d(out_channels))
            # fpn_conv = Sequential(ops.Conv2d(out_channels, out_channels, 3, padding=1), ops.BatchNorm2d(out_channels))
            self.lateral_convs.append(ops.Conv2d(in_channel, out_channels, 1, bias=False))
            self.fpn_convs.append(ops.Conv2d(out_channels, out_channels, 3, padding=1, bias=False))

    def call(self, inputs):
        """Forward compute.

        :param inputs: input feature map
        :return: tuple of feature map
        """
        laterals = [conv(inputs[i]) for i, conv in enumerate(self.lateral_convs)]
        num_stage = len(laterals)
        for i in range(num_stage - 1, 0, -1):
            laterals[i - 1] += ops.InterpolateScale(size=laterals[i - 1].size()[2:], mode='nearest')(laterals[i])
        outs = [self.fpn_convs[i](laterals[i]) for i in self.code or range(num_stage)]
        outs.append(ops.MaxPool2d(1, stride=2)(outs[-1]))
        return {idx: out for idx, out in enumerate(outs)}

    def load_state_dict(self, state_dict=None, strict=None):
        """Load state dict."""
        own_states = self.state_dict()
        not_swap_keys = []
        for own_key, own_state in own_states.items():
            state = state_dict.get("{}.{}".format(self.weights_prefix, own_key))
            if state is None or own_state.shape != state.shape:
                if 'num_batches_tracked' in own_key:
                    continue
                not_swap_keys.append(own_key)
            else:
                own_states[own_key] = state
        super().load_state_dict(own_states, strict)
