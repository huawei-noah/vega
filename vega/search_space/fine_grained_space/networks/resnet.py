# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.fine_grained_space import FineGrainedSpace
from vega.search_space.utils import get_search_space
from vega.search_space.fine_grained_space.blocks import InitialBlock, SmallInputInitialBlock
from vega.search_space.fine_grained_space.operators import op, View
from vega.search_space.fine_grained_space.cells import VariantLayer, BasicLayer


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ResNet(FineGrainedSpace):
    """Create ResNet SearchSpace."""

    _block_setting = {18: ('BasicBlock', [2, 2, 2, 2]),
                      34: ('BasicBlock', [3, 4, 6, 3]),
                      50: ('BottleneckBlock', [3, 4, 6, 3]),
                      101: ('BottleneckBlock', [3, 4, 23, 3])}

    def constructor(self, depth, block=None, items=None, num_class=10, num_reps=4, init_plane=64,
                    out_plane=512, **kwargs):
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
        self.init_block = InitialBlock(init_plane=init_plane)
        self.layers = BasicLayer(init_plane, block_cls.expansion, block, num_reps, items)
        self.AdaptiveAvgPool2d = op.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        self.Linear = op.Linear(in_features=out_plane * block_cls.expansion, out_features=num_class)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ResNetVariant(FineGrainedSpace):
    """Create ResNetVariant SearchSpace."""

    _block_setting = {18: ('BasicBlock', 8),
                      34: ('BasicBlock', 16),
                      50: ('BottleneckBlock', 16),
                      101: ('BottleneckBlock', 33)}

    def constructor(self, base_depth, base_channel, doublechannel, downsample, small_input=True, num_class=10):
        """Create layers.

        :param base_depth: base_depth
        :type num_reqs: int
        :param base_channel: base_channel
        :type items: int
        :param doublechannel: doublechannel position, 1 for use downsample, 0 for not.
        :type doublechannel: list of (0,1)
        :param downsample: downsample position, 1 for use downsample, 0 for not.
        :type downsample: list of (0,1)
        :param small_input: choose init_block.
        :type small_input: bool
        :param num_class: number of class
        :type num_class: int
        """
        if small_input:
            self.init_block = SmallInputInitialBlock(base_channel)
        else:
            self.init_block = InitialBlock(base_channel)
        if base_depth in self._block_setting.keys():
            block = dict()
            block['type'] = self._block_setting[base_depth][0]
            block_cls = get_search_space(self._block_setting[base_depth][0])
        else:
            raise ValueError("Can not find base_depth!")
        out_channel = base_channel * 2 ** sum(doublechannel) * block_cls.expansion
        self.layers = VariantLayer(base_channel, out_channel, doublechannel, downsample, block_cls.expansion, block)
        self.adaptpool = op.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        self.Linear = op.Linear(in_features=512 * block_cls.expansion, out_features=num_class)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ResNeXt(FineGrainedSpace):
    """Create ResNeXt SearchSpace."""

    _block_setting = {'resnet50_32x4d': ('BottleneckBlock', [3, 4, 6, 3], 4),
                      'resnext101_32x8d': ('BottleneckBlock', [3, 4, 23, 3], 8)}

    def constructor(self, name, block=None, items=None, num_class=10, num_reps=4, groups=None, base_width=None):
        """Create layers.

        :param base_depth: name
        :type base_depth: string
        :param base_channel: base_channel
        :type base_channel: int
        :param doublechannel: doublechannel position, 1 for use downsample, 0 for not.
        :type doublechannel: list of (0,1)
        :param downsample: downsample position, 1 for use downsample, 0 for not.
        :type downsample: list of (0,1)
        :param small_input: choose init_block.
        :type small_input: bool
        :param num_class: number of class
        :type num_class: int
        """
        if name in self._block_setting.keys():
            block = dict()
            block['type'] = self._block_setting[name][0]
            block_cls = get_search_space(self._block_setting[name][0])
            num_reps = self._block_setting[name][1]
            base_width = self._block_setting[name][2]
            groups = 32
        else:
            block_cls = get_search_space(block.get('type'))
        self.init_block = InitialBlock(init_plane=64)
        self.layers = BasicLayer(64, block_cls.expansion, block, num_reps, items, groups, base_width)
        self.AdaptiveAvgPool2d = op.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        self.Linear = op.Linear(in_features=512 * block_cls.expansion, out_features=num_class)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ResNeXtVariant(FineGrainedSpace):
    """Create ResNeXtVariant SearchSpace."""

    _block_setting = {'resnet50_32x4d': ('BottleneckBlock', 16, 4),
                      'resnext101_32x8d': ('BottleneckBlock', 33, 8)}

    def constructor(self, name, base_channel, doublechannel, downsample, small_input=True, num_class=10):
        """Create layers.

        :param base_depth: name
        :type base_depth: string
        :param base_channel: base_channel
        :type base_channel: int
        :param doublechannel: doublechannel position, 1 for use downsample, 0 for not.
        :type doublechannel: list of (0,1)
        :param downsample: downsample position, 1 for use downsample, 0 for not.
        :type downsample: list of (0,1)
        :param small_input: choose init_block.
        :type small_input: bool
        :param num_class: number of class
        :type num_class: int
        """
        if small_input:
            self.init_block = SmallInputInitialBlock(base_channel)
        else:
            self.init_block = InitialBlock(base_channel)
        if name in self._block_setting.keys():
            block = dict()
            block['type'] = self._block_setting[name][0]
            block_cls = get_search_space(self._block_setting[name][0])
            base_width = self._block_setting[name][1]
            groups = 32
        else:
            raise ValueError("Can not find base_depth!")
        out_channel = base_channel * 2 ** sum(doublechannel) * block_cls.expansion
        self.layers = VariantLayer(base_channel, out_channel, doublechannel, downsample, block_cls.expansion,
                                   block, groups, base_width)
        self.adaptpool = op.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = View()
        self.Linear = op.Linear(in_features=512 * block_cls.expansion, out_features=num_class)
