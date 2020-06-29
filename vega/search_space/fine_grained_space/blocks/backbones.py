# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for backbones."""
from vega.search_space.fine_grained_space.fine_grained_space import FineGrainedSpace
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space.blocks import InitialBlock
from vega.search_space.fine_grained_space.conditions import Append, Sequential, Map, Tuple
from vega.search_space.fine_grained_space.operators import op
from vega.search_space.utils import get_search_space


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ResNetDet(FineGrainedSpace):
    """Create ResNet_Det SearchSpace.

    As the backbone of the faster-RCNN inspection network, the fully connected layer is removed,
    the freeze part of the layer is supported, and the output of multiple layers is supported as input to the neck.
    """

    _block_setting = {18: ('BasicBlock', [2, 2, 2, 2]),
                      34: ('BasicBlock', [3, 4, 6, 3]),
                      50: ('BottleneckBlock', [3, 4, 6, 3]),
                      101: ('BottleneckBlock', [3, 4, 23, 3])}

    def constructor(self, depth, block=None, items=None, num_class=10, num_reps=4, frozen_stages=-1):
        """Create layers.

        :param num_reps: number of layers
        :type num_reqs: int
        :param items: channel and stride of every layer
        :type items: dict
        :param num_class: number of class
        :type num_class: int
        """
        if depth in self._block_setting.keys():
            block = dict()
            block['type'] = self._block_setting[depth][0]
            block_cls = get_search_space(self._block_setting[depth][0])
            num_reps = self._block_setting[depth][1]
        else:
            block_cls = get_search_space(block.get('type'))
        self.init_block = InitialBlock(init_plane=64)
        blocks = []
        in_planes = 64
        for i, num_blocks in enumerate(num_reps):
            seq = Sequential()
            out_planes = 64 * 2 ** i
            stride = 1 if i == 0 else 2
            seq.add(block_cls(inchannel=in_planes, outchannel=out_planes, groups=1, stride=stride, base_width=64))
            in_planes = out_planes * block_cls.expansion
            for idx in range(1, num_blocks):
                seq.add(block_cls(inchannel=in_planes, outchannel=out_planes, groups=1, stride=1, base_width=64))
            if i == frozen_stages:
                seq.freeze(True)
            blocks.append(seq)
        self.blocks = Append(*tuple(blocks))


@ClassFactory.register(ClassType.SEARCH_SPACE)
class RPNHead(FineGrainedSpace):
    """RpnHead."""

    def constructor(self, in_channels=256, feat_channels=256, num_classes=2):
        """Create rpn Search Space."""
        anchor_scales = [8, 16, 32]
        anchor_ratios = [0.5, 1.0, 2.0]
        num_anchors = len(anchor_ratios) * len(anchor_scales)
        if feat_channels > 0:
            conv = op.Conv2d(in_channels=in_channels, out_channels=feat_channels, kernel_size=3, padding=1)
            relu = op.ReLU(inplace=True)
            rpn_cls_conv = op.Conv2d(in_channels=feat_channels, out_channels=num_anchors * num_classes, kernel_size=1)
            rpn_reg_conv = op.Conv2d(in_channels=feat_channels, out_channels=num_anchors * 4, kernel_size=1)
            rpn_cls = Sequential(conv, relu, rpn_cls_conv)
            rpn_reg = Sequential(conv, relu, rpn_reg_conv)
        else:
            rpn_cls = op.Conv2d(in_channels=in_channels, out_channels=num_anchors * num_classes, kernel_size=1)
            rpn_reg = op.Conv2d(in_channels=in_channels, out_channels=num_anchors * 4, kernel_size=1)
        self.rpn = Tuple(Map(rpn_cls), Map(rpn_reg))
