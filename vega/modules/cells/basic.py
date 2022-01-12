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
from vega.common import ClassType, ClassFactory
from vega.modules.connections import Repeat
from vega.modules.module import Module


@ClassFactory.register(ClassType.NETWORK)
class VariantCell(Module):
    """Create VariantLayer SearchSpace."""

    def __init__(self, in_plane, out_plane, doublechannel, downsample, expansion, block, groups=1, base_width=64):
        """Create layers."""
        super(VariantCell, self).__init__()
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
        self.layers = Repeat(num_reps=num_reps, items=items, ref=block)


@ClassFactory.register(ClassType.NETWORK)
class BasicCell(Module):
    """Create BasicLayer SearchSpace."""

    def __init__(self, in_plane, expansion, block, layer_reps, items=None, groups=1, base_width=64):
        """Create layers."""
        super(BasicCell, self).__init__()
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
        self.layers = Repeat(num_reps=num_reps, items=items, ref=block)
