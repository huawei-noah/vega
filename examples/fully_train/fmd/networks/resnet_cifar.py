# -*- coding: utf-8 -*-
"""ResNet models for cifar10."""
import math
import torch.nn as nn
from vega.networks.pytorch.ops.fmdunit import FMDUnit, LinearScheduler
from vega.modules.module import Module


def conv1x1(in_plane, out_plane, stride=1):
    """1x1 convolutional layer.

    :param in_plane: size of input plane
    :type in_plane: int
    :param out_plane: size of output plane
    :type out_plane: int
    :param stride: stride of convolutional layers, default 1
    :type stride: int
    :return: Conv2d value.
    :rtype: Tensor
    """
    return nn.Conv2d(in_plane, out_plane,
                     kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding.

    :param in_plane: size of input plane
    :type in_plane: int
    :param out_plane: size of output plane
    :type out_plane: int
    :param stride: stride of convolutional layers, default 1
    :type stride: int
    :return: Conv2d value.
    :rtype: Tensor
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def linear(in_features, out_features):
    """Linear.

    :param in_features: size of input feature
    :type in_features: int
    :param out_features: size of output feature
    :type out_features: int
    :return: Linear value.
    :rtype: Tensor
    """
    return nn.Linear(in_features, out_features)


class BasicBlock(nn.Module):
    """Base module for PreResNet on small data sets.

    :param in_plane: size of input plane
    :type in_plane: int
    :param out_plane: size of output plane
    :type out_plane: int
    :param stride: stride of convolutional layers, default 1
    :type stride: int
    :param downsample: down sample type for expand dimension of input feature maps, default None
    :type downsample: bool
    :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
    of the first segment, half-preact: short cut start from the position between the first segment and the second
    one. default: both-preact
    :type block_type: str
    """

    def __init__(self, in_plane, out_plane, stride=1,
                 downsample=None, args=None, drop_prob=0.1, block_type="both_preact"):
        """Init module and weights."""
        super(BasicBlock, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.block_index = 0

    def forward(self, x):
        """Forward procedure of residual module."""
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)
        if self.downsample:
            residual = self.downsample(residual)
        out = x + residual
        return out


class BasicBlock_fmd(nn.Module):
    """Base module for PreResNet on small data sets.

    :param in_plane: size of input plane
    :type in_plane: int
    :param out_plane: size of output plane
    :type out_plane: int
    :param stride: stride of convolutional layers, default 1
    :type stride: int
    :param downsample: down sample type for expand dimension of input feature maps, default None
    :type downsample: bool
    :param block_type: type of blocks, decide position of short cut, both-preact: short cut start from beginning
    of the first segment, half-preact: short cut start from the position between the first segment and the second
    one. default: both-preact
    :type block_type: str
    """

    def __init__(self, in_plane, out_plane, stride=1,
                 downsample=None, args=None, drop_prob=0.1, block_type="both_preact"):
        """Init module and weights."""
        super(BasicBlock_fmd, self).__init__()
        self.name = block_type
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(in_plane)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_plane, out_plane, stride)
        self.dropblock1 = LinearScheduler(FMDUnit(drop_prob=drop_prob, block_size=args.block_size, args=args),
                                          start_value=0., stop_value=drop_prob, nr_steps=5e3)
        self.bn2 = nn.BatchNorm2d(out_plane)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_plane, out_plane)
        self.dropblock2 = LinearScheduler(FMDUnit(drop_prob=drop_prob, block_size=args.block_size, args=args),
                                          start_value=0., stop_value=drop_prob, nr_steps=5e3)
        self.block_index = 0

    def forward(self, x):
        """Forward procedure of residual module."""
        if self.name == "half_preact":
            x = self.bn1(x)
            x = self.relu1(x)
            residual = x
            x = self.conv1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)
        elif self.name == "both_preact":
            residual = x
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv1(x)
            x = self.dropblock1(x)
            x = self.bn2(x)
            x = self.relu2(x)
            x = self.conv2(x)
            x = self.dropblock2(x)
        if self.downsample:
            residual = self.downsample(residual)
        out = x + residual
        return out


class resnet_cifar(Module):
    """Define PreResNet on small data sets.

    :param depth: depth of network
    :type depth: int
    :param wide_factor: wide factor for deciding width of network, default is 1
    :type wide_factor: int
    :param num_classes: number of classes, related to labels. default 10
    :type num_classes: int
    """

    def __init__(self, depth, wide_factor=1, num_classes=10, args=None):
        super(resnet_cifar, self).__init__()
        self.in_plane = 16 * wide_factor
        self.depth = depth
        n = (depth - 2) / 6
        self.conv = conv3x3(3, 16 * wide_factor)
        self.layer1 = self._make_layer(BasicBlock, 16 * wide_factor, n, args=args)
        self.layer2 = self._make_layer(BasicBlock, 32 * wide_factor, n, stride=2, args=args)
        self.layer3 = self._make_layer(BasicBlock_fmd, 64 * wide_factor, n,
                                       stride=2, args=args, drop_prob=args.drop_prob)
        self.bn = nn.BatchNorm2d(64 * wide_factor)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = linear(64 * wide_factor, num_classes)
        self._init_weight()
        namlist = []
        modulelist = []
        for name, m in self.named_modules():
            namlist.append(name)
            modulelist.append(m)
        num_module = len(modulelist)
        self.dploc = []
        self.convloc = []
        for idb in range(num_module):
            if isinstance(modulelist[idb], FMDUnit):
                self.dploc.append(idb)
                for iconv in range(idb, num_module):
                    if isinstance(modulelist[iconv], nn.Conv2d) and not ('downsample' in namlist[iconv]):
                        self.convloc.append(iconv)
                        break

    def _init_weight(self):
        """Init layer parameters."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_plane, n_blocks, stride=1, args=None, drop_prob=0.1):
        """Make residual blocks, including short cut and residual function.

        :param block: type of basic block to build network
        :type block: nn.Module
        :param out_plane: size of output plane
        :type out_plane: int
        :param n_blocks: number of blocks on every segment
        :type n_blocks: int
        :param stride: stride of convolutional neural network, default 1
        :type stride: int
        :return: residual blocks
        :rtype: nn.Sequential
        """
        downsample = None
        if stride != 1 or self.in_plane != out_plane:
            downsample = conv1x1(self.in_plane, out_plane, stride=stride)
        layers = []
        layers.append(block(self.in_plane, out_plane, stride,
                            downsample, block_type="both_preact", args=args, drop_prob=drop_prob))
        self.in_plane = out_plane
        for i in range(1, int(n_blocks)):
            layers.append(block(self.in_plane, out_plane, args=args, drop_prob=drop_prob))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward procedure of model."""
        modulelist = list(self.modules())
        for imodu in range(len(self.convloc)):
            modulelist[self.dploc[imodu]].weight_behind = modulelist[self.convloc[imodu]].weight.data

        for module in self.modules():
            if isinstance(module, LinearScheduler):
                module.step()

        out = self.conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
