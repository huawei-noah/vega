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

"""DNet network."""
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.operators import ops
from vega.modules.connections import Sequential, ModuleList


@ClassFactory.register(ClassType.NETWORK)
class DNet(Module):
    """Dnet network."""

    def __init__(self, encoding, n_class=1000):
        super().__init__()
        self.backbone = DNetBackbone(encoding)
        self.view = ops.View()
        out_plane = self.backbone.out_channels
        self.fc = ops.Linear(in_features=out_plane, out_features=n_class)

    def call(self, x):
        """Call DNet backbone."""
        x = self.backbone(x)
        x = self.view(x)
        return self.fc(x)


@ClassFactory.register(ClassType.NETWORK)
class DNetBackbone(Module):
    """DNet backbone network."""

    def __init__(self, encoding):
        super(DNetBackbone, self).__init__()
        op_names = ["conv3", "conv1", "conv3_grp2", "conv3_grp4", "conv3_base1", "conv3_base32", "conv3_sep"]

        # code with kangning
        block_str, num_channel, macro_str = encoding.split('_')
        curr_channel, index = int(num_channel), 0

        _big_model = "*" in block_str
        if _big_model:
            block_encoding_list = block_str.split('*')

        # stem
        layers = [
            create_op('conv3', 3, curr_channel // 2, stride=2),
            ops.Relu(),
            create_op('conv3', curr_channel // 2, curr_channel // 2),
            ops.Relu(),
            create_op('conv3', curr_channel // 2, curr_channel, stride=2),
            ops.Relu()
        ]

        # body
        if not _big_model:
            while index < len(macro_str):
                stride = 1
                if macro_str[index] == '-':
                    stride = 2
                    index += 1

                channel_increase = int(macro_str[index])
                block = EncodedBlock(block_str, curr_channel, op_names, stride, channel_increase)
                layers.append(block)
                curr_channel *= channel_increase
                index += 1
        else:
            block_encoding_index = 0
            while index < len(macro_str):
                stride = 1
                if macro_str[index] == '-':
                    stride = 2
                    index += 1
                    block_encoding_index += 1
                channel_increase = int(macro_str[index])
                block_encoding = block_encoding_list[block_encoding_index]
                block = EncodedBlock(block_encoding, curr_channel, op_names, stride, channel_increase)
                layers.append(block)
                curr_channel *= channel_increase
                index += 1
        layers.append(ops.AdaptiveAvgPool2d((1, 1)))
        self.layers = Sequential(*layers)

    def call(self, x, **kwargs):
        """Implement function all."""
        return self.layers(x)


def conv33(in_channel, out_channel, stride=1, groups=1, bias=False):
    """Conv 3*3."""
    if groups != 0 and in_channel % groups != 0:
        raise ValueError('In channel "{}" is not a multiple of groups: "{}"'.format(
            in_channel, groups))
    if out_channel % groups != 0:
        raise ValueError('Out channel "{}" is not a multiple of groups: "{}"'.format(
            out_channel, groups))

    return ops.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3,
                      stride=stride, bias=bias)


def conv11(in_channel, out_channel, stride=1, bias=False):
    """Conv 1*1."""
    return ops.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1,
                      stride=stride, bias=bias)


def conv33_base(in_channel, out_channel, stride=1, base_channel=1):
    """Conv 3*3."""
    return conv33(in_channel, out_channel, stride, in_channel // base_channel)


def conv33_sep(in_channel, out_channel, stride):
    """Conv 3*3 sep."""
    return Sequential(
        conv33(in_channel, in_channel, stride, groups=in_channel),
        conv11(in_channel, out_channel))


OPS = {
    'conv3': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride),
    'conv1': lambda in_channel, out_channel, stride: conv11(in_channel, out_channel, stride),
    'conv3_grp2': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride, groups=2),
    'conv3_grp4': lambda in_channel, out_channel, stride: conv33(in_channel, out_channel, stride, groups=4),
    'conv3_base1': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride, base_channel=1),
    # noqa: E501
    'conv3_base16': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride,
                                                                        base_channel=16),  # noqa: E501
    'conv3_base32': lambda in_channel, out_channel, stride: conv33_base(in_channel, out_channel, stride,
                                                                        base_channel=32),  # noqa: E501
    'conv3_sep': lambda in_channel, out_channel, stride: conv33_sep(in_channel, out_channel, stride)
}


def create_op(opt_name, in_channel, out_channel, stride=1):
    """Create op."""
    layer = OPS[opt_name](in_channel, out_channel, stride)
    bn = ops.BatchNorm2d(out_channel)
    return Sequential(layer, bn)


class AddBlock(Module):
    """Add Block."""

    def __init__(self, layer_sizes, strides, num1, num2):
        super(AddBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None
        stride = 1
        if strides[num1] != strides[num2]:
            stride = 2
        if stride != 1 or layer_sizes[num1] != layer_sizes[num2]:
            self.conv = create_op('conv1', layer_sizes[num1], layer_sizes[num2], stride)

    def call(self, x, **kwargs):
        """call."""
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = x1 + x2
        return x


class ConcatBlock(Module):
    """Concat Block."""

    def __init__(self, layer_sizes, strides, num1, num2):
        super(ConcatBlock, self).__init__()
        self.num1 = num1
        self.num2 = num2
        self.conv = None
        stride = 1
        if strides[num1] != strides[num2]:
            stride = 2
        if stride != 1:
            self.conv = create_op('conv1', layer_sizes[num1], layer_sizes[num1], stride)
        layer_sizes[self.num2] += layer_sizes[self.num1]

    def call(self, x, **kwargs):
        """call."""
        x1, x2 = x[self.num1], x[self.num2]
        if self.conv is not None:
            x1 = self.conv(x1)
        x[self.num2] = ops.concat([x1, x2], 1)
        return x


class EncodedBlock(Module):
    """Encode block."""

    def __init__(self, block_str, in_channel, op_names, stride=1, channel_increase=1):
        super(EncodedBlock, self).__init__()

        if "-" in block_str:
            layer_str, connect_str = block_str.split('-')
        else:
            layer_str, connect_str = block_str, ""

        layer_str = layer_str + "2"
        base_channel = in_channel * channel_increase
        layer_sizes = [in_channel]
        connect_parts = [connect_str[i:i + 3] for i in range(0, len(connect_str), 3)]
        connect_parts = sorted(connect_parts, key=lambda x: x[2])
        connect_index = 0

        self.module_list = ModuleList()
        length = len(layer_str) // 2
        stride_place = 0
        while (stride_place + 1) * 2 < len(layer_str) and layer_str[stride_place * 2] == '1':
            stride_place += 1

        strides = [1] * (stride_place + 1) + [stride] * (length - stride_place)
        connect_parts.append("a0{}".format(length))

        for i in range(length):
            layer_module_list = ModuleList()
            layer_opt_name = op_names[int(layer_str[i * 2])]
            layer_in_channel = layer_sizes[-1]
            layer_out_channel = base_channel * 2 ** int(layer_str[i * 2 + 1]) // 4
            layer_stride = stride if i == stride_place else 1
            layer = create_op(layer_opt_name, layer_in_channel, layer_out_channel, layer_stride)
            layer_module_list.append(layer)
            layer_sizes.append(layer_out_channel)

            while connect_index < len(connect_parts) and int(connect_parts[connect_index][2]) == i + 1:
                block_class = AddBlock if connect_parts[connect_index][0] == 'a' else ConcatBlock
                block = block_class(
                    layer_sizes, strides, int(connect_parts[connect_index][1]), int(connect_parts[connect_index][2]))
                layer_module_list.append(block)
                connect_index += 1

            self.module_list.append(layer_module_list)
        self.tmp1 = []
        self.tmp3 = []
        for index, tmp in enumerate(self.module_list):
            self.tmp1.append(tmp)
            tmp2 = []
            for i, layer in enumerate(tmp):
                tmp2.append(layer)
            self.tmp3.append(tmp2)

    def call(self, x, **kwargs):
        """call."""
        outs = [x]
        current = x

        for index, module_layer in enumerate(self.tmp1):
            for i, layer in enumerate(self.tmp3[index]):
                if i == 0:
                    outs.append(layer(current))
                else:
                    outs = layer(outs)
            current = ops.Relu()(outs[-1])

        return current
