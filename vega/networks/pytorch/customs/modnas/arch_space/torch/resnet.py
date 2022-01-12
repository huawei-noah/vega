# -*- coding:utf-8 -*-

# This file is adapted from the torchvision library at
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

# 2020.6.29-Changed for Modular-NAS search space.
#         Huawei Technologies Co., Ltd. <linyunfeng5@huawei.com>
# Copyright 2020 Huawei Technologies Co., Ltd.

"""ResNet architectures."""

from functools import partial
import torch.nn as nn
from modnas.registry.construct import DefaultSlotTraversalConstructor
from modnas.registry.construct import register as register_constructor
from modnas.registry.arch_space import register
from ..ops import Identity
from ..slot import Slot


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """Return 3x3 convolution with padding."""
    return Slot(_chn_in=in_planes, _chn_out=out_planes, _stride=stride, groups=groups)


def conv1x1(in_planes, out_planes, stride=1):
    """Return 1x1 convolution."""
    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_planes))


class BasicBlock(nn.Module):
    """Basic Block class."""

    expansion = 1
    chn_init = 16

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        del base_width
        self.conv1 = conv3x3(inplanes, planes, stride, groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Compute network output."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck block class."""

    expansion = 4
    chn_init = 16

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=None, norm_layer=None):
        super(Bottleneck, self).__init__()
        width = int(planes * (1. * base_width / self.chn_init)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.conv2 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """Compute network output."""
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet architecture class."""

    def __init__(self,
                 chn_in,
                 chn,
                 block,
                 layers,
                 n_classes,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=None,
                 use_bn=False,
                 expansion=None):
        super(ResNet, self).__init__()
        if use_bn:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = Identity
        self.use_bn = use_bn
        if expansion is not None:
            block.expansion = expansion
        block.chn_init = chn

        self.chn = chn
        self.groups = groups
        self.base_width = chn // groups if width_per_group is None else width_per_group
        self.conv1 = self.get_stem(chn_in, chn, nn.BatchNorm2d)

        self.layers = nn.Sequential(*[
            self._make_layer(block, (2**i) * chn, layers[i], stride=(1 if i == 0 else 2), norm_layer=norm_layer)
            for i in range(len(layers))
        ])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.chn, n_classes)
        self.zero_init_residual = zero_init_residual

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None):
        downsample = None
        if stride != 1 or self.chn != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(
                self.chn,
                planes * block.expansion,
                stride,
            ), )

        layers = []
        layers.append(block(self.chn, planes, stride, downsample, self.groups, self.base_width, norm_layer=norm_layer))
        self.chn = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.chn, planes, 1, None, self.groups, self.base_width, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        """Compute network output."""
        x = self.conv1(x)

        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


@register_constructor
class ResNetPredefinedConstructor(DefaultSlotTraversalConstructor):
    """ResNet original network constructor."""

    def __init__(self, use_bn=False):
        super().__init__()
        self.use_bn = use_bn

    def convert(self, slot):
        """Convert slot to module."""
        return nn.Sequential(
            nn.Conv2d(slot.chn_in, slot.chn_out, 3, stride=slot.stride, padding=1, bias=False, **slot.kwargs),
            nn.BatchNorm2d(slot.chn_out) if self.use_bn else Identity(),
        )


class ImageNetResNet(ResNet):
    """ResNet for ImageNet dataset."""

    def get_stem(self, chn_in, chn, norm_layer):
        """Return stem layers."""
        return nn.Sequential(
            nn.Conv2d(chn_in, chn, kernel_size=7, stride=2, padding=3, bias=False),
            norm_layer(chn),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )


class CIFARResNet(ResNet):
    """ResNet for CIFAR dataset."""

    def get_stem(self, chn_in, chn, norm_layer):
        """Return stem layers."""
        return nn.Sequential(
            nn.Conv2d(chn_in, chn, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(chn),
            nn.ReLU(inplace=False),
        )


def resnet10(resnet_cls, **kwargs):
    """Construct a ResNet-10 model."""
    return resnet_cls(block=BasicBlock, layers=[1, 1, 1, 1], **kwargs)


def resnet18(resnet_cls, **kwargs):
    """Construct a ResNet-18 model."""
    return resnet_cls(block=BasicBlock, layers=[2, 2, 2, 2], **kwargs)


def resnet32(resnet_cls, **kwargs):
    """Construct a ResNet-32 model."""
    return resnet_cls(block=BasicBlock, layers=[5, 5, 5], **kwargs)


def resnet34(resnet_cls, **kwargs):
    """Construct a ResNet-34 model."""
    return resnet_cls(block=BasicBlock, layers=[3, 4, 6, 3], **kwargs)


def resnet50(resnet_cls, **kwargs):
    """Construct a ResNet-50 model."""
    return resnet_cls(block=Bottleneck, layers=[3, 4, 6, 3], **kwargs)


def resnet56(resnet_cls, **kwargs):
    """Construct a ResNet-56 model."""
    return resnet_cls(block=BasicBlock, layers=[9, 9, 9], **kwargs)


def resnet101(resnet_cls, **kwargs):
    """Construct a ResNet-101 model."""
    return resnet_cls(block=Bottleneck, layers=[3, 4, 23, 3], **kwargs)


def resnet110(resnet_cls, **kwargs):
    """Construct a ResNet-110 model."""
    return resnet_cls(block=BasicBlock, layers=[18, 18, 18], **kwargs)


def resnet152(resnet_cls, **kwargs):
    """Construct a ResNet-152 model."""
    return resnet_cls(block=Bottleneck, layers=[3, 8, 36, 3], **kwargs)


def resnext50_32x4d(resnet_cls, **kwargs):
    """Construct a ResNeXt-50 32x4d model."""
    return resnet_cls(block=Bottleneck, layers=[3, 4, 6, 3], groups=32, width_per_group=4, **kwargs)


def resnext101_32x8d(resnet_cls, **kwargs):
    """Construct a ResNeXt-50 32x8d model."""
    return resnet_cls(block=Bottleneck, layers=[3, 4, 23, 3], groups=32, width_per_group=8, **kwargs)


def resnet(resnet_cls, bottleneck=False, **kwargs):
    """Construct a ResNet model."""
    block = Bottleneck if bottleneck else BasicBlock
    return resnet_cls(block=block, **kwargs)


for net_cls in [CIFARResNet, ImageNetResNet]:
    name = 'CIFAR-' if net_cls == CIFARResNet else 'ImageNet-'
    register(partial(resnet10, net_cls), name + 'ResNet-10')
    register(partial(resnet18, net_cls), name + 'ResNet-18')
    register(partial(resnet32, net_cls), name + 'ResNet-32')
    register(partial(resnet34, net_cls), name + 'ResNet-34')
    register(partial(resnet50, net_cls), name + 'ResNet-50')
    register(partial(resnet56, net_cls), name + 'ResNet-56')
    register(partial(resnet101, net_cls), name + 'ResNet-101')
    register(partial(resnet152, net_cls), name + 'ResNet-152')
    register(partial(resnext50_32x4d, net_cls), name + 'ResNeXt-50')
    register(partial(resnext101_32x8d, net_cls), name + 'ResNeXt-101')
    register(partial(resnet, net_cls), name + 'ResNet')
