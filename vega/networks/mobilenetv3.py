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

"""This is SearchSpace for mobilenetv3."""
import math
from vega.common import ClassFactory, ClassType
from vega.modules.connections import Sequential
from vega.modules.module import Module
from vega.modules.operators import ops


def _make_divisible(v, divisor, min_value=None):
    """Make value divisible."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


@ClassFactory.register(ClassType.NETWORK)
class SELayer(Module):
    """This is the class of Squeeze-and-Excite layer for MobileNetV3."""

    def __init__(self, channel, reduction=4):
        """Init SELayer."""
        super(SELayer, self).__init__()
        self.avg_pool = ops.AdaptiveAvgPool2d(1)
        hidden_dim = _make_divisible(channel // reduction, 8)
        self.fc = Sequential(
            ops.Linear(channel, hidden_dim, use_bias=False),
            ops.Relu(inplace=True),
            ops.Linear(hidden_dim, channel, use_bias=False),
            ops.Hsigmoid()
        )

    def __call__(self, x):
        """Forward compute of SELayer."""
        b, c, _, _ = x.shape
        y = ops.View((b, c))(self.avg_pool(x))
        y = ops.View((b, c, 1, 1))(self.fc(y))
        return x * y


@ClassFactory.register(ClassType.NETWORK)
class ConvBnAct(ops.Module):
    """Create group of Convolution + BN + Activation."""

    def __init__(self, C_in, C_out, kernel_size, stride, padding, bias=False, momentum=0.1,
                 affine=True, activation='relu', inplace=True):
        """Construct ConvBnAct class."""
        super(ConvBnAct, self).__init__()
        self.conv2d = ops.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=bias)
        self.batch_norm2d = ops.BatchNorm2d(C_out, affine=affine, momentum=momentum)
        if activation == 'hswish':
            self.act = ops.Hswish(inplace=inplace)
        elif activation == 'hsigmoid':
            self.act = ops.Hsigmoid(inplace=inplace)
        elif activation == 'relu6':
            self.act = ops.Relu6(inplace=inplace)
        else:
            self.act = ops.Relu(inplace=inplace)


@ClassFactory.register(ClassType.NETWORK)
class InvertedResidualSE(Module):
    """This is the class of InvertedResidual with SELayer for MobileNetV3."""

    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se=False, use_hs=False, momentum=0.1):
        """Init InvertedResidualSE."""
        super(InvertedResidualSE, self).__init__()
        self.identity = stride == 1 and inp == oup
        self.ir_block = Sequential(
            # pw
            ops.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            ops.BatchNorm2d(hidden_dim, momentum=momentum),
            ops.Hswish() if use_hs else ops.Relu(inplace=True),
            # dw
            ops.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2,
                       groups=hidden_dim, bias=False),
            ops.BatchNorm2d(hidden_dim, momentum=momentum),
            # Squeeze-and-Excite
            SELayer(hidden_dim) if use_se else Sequential(),
            ops.Hswish() if use_hs else ops.Relu(inplace=True),
            # pw-linear
            ops.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            ops.BatchNorm2d(oup, momentum=momentum),
        )

    def __call__(self, x):
        """Forward compute of InvertedResidualSE."""
        if self.identity:
            return x + self.ir_block(x)
        else:
            return self.ir_block(x)


class MobileNetV3(Module):
    """This is the base class of MobileNetV3."""

    def __init__(self, cfgs, mode='small', input_channel=3, feat_channels=16, special_stride=1, num_classes=10,
                 width_mult=1., block=InvertedResidualSE, momentum=0.1, is_prune_mode=False, **kwargs):
        """Init MobileNetV3.

        :params cfgs: cfgs for mobilenetv3
        :type cfgs: list
        :params special_stride: the stride of the first InvertedResidualSE block.
        :type special_stride: int (1 for cifar10, 2 for imagenet)
        """
        super(MobileNetV3, self).__init__()
        self.cfgs = cfgs

        # building first layer
        if not is_prune_mode:
            feat_channels = _make_divisible(feat_channels * width_mult, 8)
        else:
            feat_channels = int(feat_channels * width_mult)
        layers = [ConvBnAct(input_channel, feat_channels, kernel_size=3, momentum=momentum,
                            stride=special_stride, padding=1, activation='hswish')]

        # buidling blocks
        # kernel_size, expand_ratio, output_channels, use_se, use_hs, stride
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8) if not is_prune_mode else int(c * width_mult)
            hidden_dim = _make_divisible(t, 8) if not is_prune_mode else t
            layers.append(block(feat_channels, hidden_dim, output_channel, k, s, use_se, use_hs, momentum))
            feat_channels = output_channel
        self.features = Sequential(*layers)

        # building last linear layer
        self.avgpool = ops.AdaptiveAvgPool2d((1, 1))
        chn = 1280 if mode == 'large' else 1024
        self.classifier = Sequential(
            ops.View(),
            ops.Linear(feat_channels, chn),
            ops.Hswish(),
            ops.Dropout(0.2),
            ops.Linear(chn, num_classes)
        )
        self._initialize_weights()

    def __call__(self, x):
        """Forward compute of MobileNetV3 for classification."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, ops.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ops.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, ops.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


@ClassFactory.register(ClassType.NETWORK)
class MobileNetV3Small(MobileNetV3):
    """Create small version MobileNetV3."""

    cfgs = [
        # k, t, c, SE, HS, s
        [3, 16, 16, 1, 0, 1],
        [3, 72, 24, 0, 0, 2],
        [3, 88, 24, 0, 0, 1],
        [5, 96, 40, 1, 1, 2],
        [5, 240, 40, 1, 1, 1],
        [5, 240, 40, 1, 1, 1],
        [5, 120, 48, 1, 1, 1],
        [5, 144, 48, 1, 1, 1],
        [5, 288, 96, 1, 1, 2],
        [5, 576, 96, 1, 1, 1],
        [5, 576, 96, 1, 1, 1],
    ]

    def __init__(self, cfgs=None, mode='small', input_channel=3, feat_channels=16, special_stride=1, num_classes=10,
                 width_mult=1., block=InvertedResidualSE, momentum=0.1, is_prune_mode=False, **kwargs):
        """Init MobileNetV3Small."""
        if cfgs is None:
            if special_stride != self.cfgs[0][-1]:
                self.cfgs[0][-1] = special_stride
            cfgs = self.cfgs
        super(MobileNetV3Small, self).__init__(cfgs, mode, input_channel, feat_channels, special_stride, num_classes,
                                               width_mult, block, momentum, is_prune_mode, **kwargs)


@ClassFactory.register(ClassType.NETWORK)
class MobileNetV3Large(MobileNetV3):
    """Create large version MobileNetV3."""

    cfgs = [
        # k, t, c, SE, HS, s
        [3, 16, 16, 0, 0, 1],
        [3, 64, 24, 0, 0, 1],
        [3, 72, 24, 0, 0, 1],
        [5, 72, 40, 1, 0, 2],
        [5, 120, 40, 1, 0, 1],
        [5, 120, 40, 1, 0, 1],
        [3, 240, 80, 0, 1, 2],
        [3, 200, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 184, 80, 0, 1, 1],
        [3, 480, 112, 1, 1, 1],
        [3, 672, 112, 1, 1, 1],
        [5, 672, 160, 1, 1, 2],
        [5, 960, 160, 1, 1, 1],
        [5, 960, 160, 1, 1, 1]
    ]

    def __init__(self, cfgs=None, mode='large', input_channel=3, feat_channels=16, special_stride=1, num_classes=10,
                 width_mult=1., block=InvertedResidualSE, momentum=0.1, is_prune_mode=False, **kwargs):
        """Init MobileNetV3Large."""
        if cfgs is None:
            if special_stride != self.cfgs[1][-1]:
                self.cfgs[0][-1] = special_stride
            cfgs = self.cfgs
        super(MobileNetV3Large, self).__init__(cfgs, mode, input_channel, feat_channels, special_stride, num_classes,
                                               width_mult, block, momentum, is_prune_mode, **kwargs)
