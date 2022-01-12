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

"""This is TextCNN network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.operators import ops
from vega.modules.connections import Concat
from vega.modules.blocks import TextConvBlock


@ClassFactory.register(ClassType.NETWORK)
class TextCells(Module):
    """Create Text Conv Cell."""

    def __init__(self, in_channels=1, embed_dim=8, out_channels=16):
        super(TextCells, self).__init__()
        self.kernels = self.define_arch_params('kernels', [3, 4, 5])
        self.cells = Concat()
        for idx, kernel_size in enumerate(self.kernels):
            kernel_size = (kernel_size, embed_dim)
            block = TextConvBlock(in_channels, out_channels, kernel_size)
            self.cells.add(block)

    @property
    def out_channels(self):
        """Output Channel for ResNet backbone."""
        last_channel = super().out_channels
        return len(self.kernels) * last_channel


@ClassFactory.register(ClassType.NETWORK)
class TextCNN(Module):
    """Create TextCNN Module."""

    def __init__(self, in_channels=1, embed_dim=8, kernel_num=16, num_class=2):
        super(TextCNN, self).__init__()
        self.cells = TextCells(in_channels, embed_dim, kernel_num)
        self.head = ops.Linear(self.cells.out_channels, num_class, activation='softmax')
