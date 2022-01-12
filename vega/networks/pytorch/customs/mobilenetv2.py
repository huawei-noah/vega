# -*- coding:utf-8 -*-

# This file is adapted from the torchvision library at
# https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py

# 2021.07.04-Changed for vega search space
# Author: Nikita Klyuchnikov <nikita.klyuchnikov@skolkovotech.ru>

"""MobileNetV2 architecture."""

import torch.nn as nn
from vega.common import ClassFactory, ClassType


def _make_divisible(v, divisor, min_value=None):
    """
    Taken from the original tf repo.

    It ensures that all layers have a channel number that is divisible by 8
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    """ConvBNReLU class."""

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        """Initialize ConvBNReLU instance."""
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    """InvertedResidual class."""

    def __init__(self, inp, oup, stride, expand_ratio, kernel_size=3):
        """Initialize InvertedResidual instance."""
        super(InvertedResidual, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError("Stride must be in [1,2].")

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, kernel_size=kernel_size),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        """Pass forward x."""
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@ClassFactory.register(ClassType.NETWORK)
class MobileNetV2(nn.Module):
    """MobileNetV2 main class."""

    def __init__(self,
                 num_classes=10,
                 width_mult=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8,
                 block=None, kernels=None, first_stride=1, last_channel=1280, desc=None):
        """
        Network MobileNet V2 main class.

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet

        """
        super(MobileNetV2, self).__init__()

        if kernels is None:
            kernels = [3] * 7
        if block is None:
            block = InvertedResidual
        input_channel = 32

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        if desc is not None:
            for i in range(len(inverted_residual_setting)):
                inverted_residual_setting[i][1] = desc['layer_%d' % i]['channels']
                inverted_residual_setting[i][2] = desc['layer_%d' % i]['repetitions']

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=first_stride)]

        idx = 0

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, kernel_size=kernels[idx]))
                input_channel = output_channel

            idx += 1
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x, is_feat):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass

        kd_layers = []

        for i in range(len(self.features)):
            x = self.features[i](x)

            if i in [0, 2, 3, 5, 7]:
                kd_layers.append(x)

        f5 = x.mean([2, 3])
        out = self.classifier(f5)

        if is_feat:
            return kd_layers, out
        else:
            return out

    def forward(self, x, is_feat=False, preact=False):
        """Pass forward x."""
        return self._forward_impl(x, is_feat)
