# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space import Repeat, FineGrainedSpace


@ClassFactory.register(ClassType.SEARCH_SPACE)
class VariantLayer(FineGrainedSpace):
    """Create VariantLayer SearchSpace."""

    def __init__(self, in_plane, out_plane, doublechannel, downsample, expansion, block, groups=1, base_width=64):
        """Create layers."""
        super(VariantLayer, self).__init__(in_plane, out_plane, doublechannel, downsample, expansion, block, groups,
                                           base_width)
        items = {}
        inchannel = []
        outchannel = []
        stride = []
        num_reps = len(doublechannel)
        out_plane = in_plane
        for i in range(num_reps):
            inchannel.append(in_plane)
            out_plane = out_plane if doublechannel[i] == 0 else out_plane * 2
            outchannel.append(out_plane)
            in_plane = out_plane * expansion
            if downsample[i] == 0:
                stride.append(1)
            else:
                stride.append(2)
        items['inchannel'] = inchannel
        items['outchannel'] = outchannel
        items['stride'] = stride
        items['groups'] = [groups] * num_reps
        items['base_width'] = [base_width] * num_reps
        self.repeat = Repeat(num_reps=num_reps, items=items, ref=block)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class BasicLayer(FineGrainedSpace):
    """Create BasicLayer SearchSpace."""

    def __init__(self, in_plane, expansion, block, layer_reps, items=None, groups=1, base_width=64):
        """Create layers.

        :param in_plane: in_plane
        :type in_plane: int
        :param expansion: expansion.
        :type expansion: int
        :param block: block
        :type block: dict
        :param layer_reps: depth of each layer.
        :type layer_reps: list
        :param items: items
        :type items: dict
        :param groups: groups, default 1.
        :type groups: int
        :param base_width: width of each group.
        :type base_width: int
        """
        super(BasicLayer, self).__init__(in_plane, expansion, block, layer_reps, items, groups, base_width)
        if items is None:
            inplane = in_plane
            items = {}
            inchannel = []
            outchannel = []
            stride = []
            for i in range(len(layer_reps)):
                outplane = 64 * (2 ** i)
                inchannel.append(inplane)
                outchannel.append(outplane)
                inplane = outplane * expansion
                stride.append(1 if i == 0 else 2)
                for _ in range(1, layer_reps[i]):
                    inchannel.append(inplane)
                    outchannel.append(outplane)
                    stride.append(1)
            items['inchannel'] = inchannel
            items['outchannel'] = outchannel
            items['stride'] = stride
            items['groups'] = [groups] * len(inchannel)
            items['base_width'] = [base_width] * len(inchannel)
            num_reps = sum(layer_reps)
            self.repeat = Repeat(num_reps=num_reps, items=items, ref=block)
        else:
            self.repeat = Repeat(num_reps=len(layer_reps) if isinstance(layer_reps, list) else layer_reps,
                                 items=items, ref=block)
