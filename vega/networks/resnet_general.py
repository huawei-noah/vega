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

"""This is SearchSpace for general network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.get_module_class import get_module_class
from vega.modules.blocks import InitialBlock, SmallInputInitialBlock
from vega.modules.connections import Repeat
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class ResNetGeneral(Module):
    """Create ResNet General SearchSpace."""

    _block_setting = {
        18: ('BasicBlock', 8),
        20: ('BasicBlock', 11),
        34: ('BasicBlock', 16),
        50: ('BottleneckBlock', 16),
        101: ('BottleneckBlock', 33),
        152: ('BottleneckBlock', 50),
    }

    _default_blocks = {
        18: [2, 2, 2, 2],
        20: [3, 3, 3, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
    }

    def __init__(self, small_input=True, base_channel=64, base_depth=50, stage=4,
                 doublechannel=None, downsample=None, **kwargs):
        """Create layers."""
        super(ResNetGeneral, self).__init__()
        self.base_channel = base_channel
        self.base_depth = base_depth
        self.stages = stage
        self.block_type = self._block_setting[self.base_depth][0]
        self.block_stage = self._default_blocks[self.base_depth][:self.stages]

        node_channels = self.prune_setting()
        self.block_cls = get_module_class(self.block_type)
        if doublechannel is None:
            doublechannel = self._generate_default_struct()
        if downsample is None:
            downsample = self._generate_default_struct()
        if node_channels is None:
            self.inchannel_list, self.outchannel_list = \
                self._generate_channel_list(self.base_channel, doublechannel)
        else:
            self.inchannel_list, self.outchannel_list = \
                self._transform_channel_list(self.base_channel, node_channels)
        self.stride_list = self._generate_stride_list(downsample)

        self.init_block = \
            SmallInputInitialBlock(self.base_channel) if small_input else InitialBlock(self.base_channel)
        ref_block = {'type': self.block_type}
        self.layers = self.resnet_cell(ref_block)

    def _generate_default_struct(self):
        """Generate default struct for doublechannel and downsample."""
        blocks = self.block_stage
        struct = []
        for i, block in enumerate(blocks):
            stage = [0] * block
            if i > 0:
                stage[0] = 1
            struct = struct + stage
        return struct

    def _generate_channel_list(self, base_channel, double_channel):
        """Generate channel list of all blocks."""
        inchannel_list = []
        outchannel_list = []
        expansion = self.block_cls.expansion
        current_in = base_channel
        current_out = base_channel
        for double in double_channel:
            if double == 1:
                current_out *= 2
            inchannel_list.append(current_in)
            outchannel_list.append(current_out)
            current_in = current_out * expansion
        return inchannel_list, outchannel_list

    def _transform_channel_list(self, base_channel, node_channels):
        """Transform node channels to channel list."""
        blocks = self.block_stage
        outchannel_list = []
        for i in range(len(node_channels)):
            outchannel_list += [node_channels[i]] * blocks[i]
        inchannel_list = []
        expansion = self.block_cls.expansion
        current_in = base_channel
        for current_out in outchannel_list:
            inchannel_list.append(current_in)
            current_in = current_out * expansion
        return inchannel_list, outchannel_list

    def _generate_stride_list(self, down_sample):
        """Generate stride list of all blocks."""
        stride_list = []
        for down in down_sample:
            stride_list.append(2 if down == 1 else 1)
        return stride_list

    def prune_setting(self):
        """Prune setting if possible."""
        node_channels = self.desc.get('chn_node', None)
        if node_channels is None:
            return None
        self.inner_channels = self.desc.get('chn', None)
        self.block_type = 'PruneBasicBlock'
        return node_channels

    def resnet_cell(self, ref_block):
        """Construct ResNet main cell."""
        items = {}
        items['inchannel'] = self.inchannel_list
        items['outchannel'] = self.outchannel_list
        items['stride'] = self.stride_list
        if hasattr(self, 'inner_channels'):
            items['innerchannel'] = self.inner_channels
        cell = Repeat(num_reps=len(self.stride_list), items=items, ref=ref_block)
        return cell

    @property
    def out_channels(self):
        """Output Channel for ResNet backbone."""
        return [module.out_channels for name, module in self.named_modules() if isinstance(module, ops.Conv2d)][-1]
