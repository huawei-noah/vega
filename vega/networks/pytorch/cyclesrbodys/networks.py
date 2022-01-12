# -*- coding: utf-8 -*-

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

"""Basic layers."""

import logging
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.nn.modules as Module
import vega


try:
    import horovod.torch as hvd
except Exception:
    logging.debug("horovod not been installed.")


def initialize(nets, init_gain=0.02, use_cuda=True, use_distributed=False):
    """Initialize a network.

    :param nets: list of networks to be initialized
    :type nets: list
    :param init_gain: scaling factor for normal, xavier and orthogonal
    :type init_gain: float
    :param use_cuda: if use cuda mode
    :type use_cuda: bool
    :return: Return an initialized network.
    :rtype: nn.Module
    """

    def init_w(module):
        classname = module.__class__.__name__
        if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(module.weight.data, 0.0, init_gain)
            if hasattr(module, 'bias') and module.bias is not None:
                init.constant_(module.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(module.weight.data, 1.0, init_gain)
            init.constant_(module.bias.data, 0.0)

    module_nets = []
    for net in nets:
        if use_cuda:
            if use_distributed:
                net = torch.nn.DataParallel(net, device_ids=[hvd.local_rank()])
            else:
                if vega.is_npu_device():
                    net = torch.nn.DataParallel(net).npu()
                else:
                    net = torch.nn.DataParallel(net).cuda()
        logging.info('==> Initialize network with normal')
        net.apply(init_w)
        module_nets.append(net)
    return module_nets


def requires_grad(nets, requires_grads=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations.

    :param nets: a list of networks
    :type nets: list
    :param requires_grads: whether the networks require gradients or not
    :type requires_grads: bool
    """
    for net in nets:
        for param in net.parameters():
            param.requires_grad = requires_grads


class ShortcutBlock(nn.Module):
    """Add shortcut connection to a submodule.

    :param module: submodule to be connected
    :type module: nn.Module
    """

    def __init__(self, module):
        super(ShortcutBlock, self).__init__()
        self.submodule = module

    def forward(self, x):
        """Forward process."""
        output = x + self.submodule(x)
        return output


class Generator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    :param in_c: the number of channels in input images
    :type in_c: int
    :param out_c: the number of channels in output images
    :type out_c: int
    :param ngf: the base number of filters in generator
    :type ngf: int
    :param n_resblocks: number of resnet blocks
    :type n_resblocks: int
    :param norm_type: normalization layer
    :type norm_type: str
    :param act_type: activation layer
    :type act_type: str
    :param up_mode: type of upsample
    :type up_mode: str
    """

    def __init__(self, in_c, out_c, ngf=64, n_resblocks=9, norm_type='batch', act_type='relu',
                 up_mode='transpose', name='G'):
        """Construct a Resnet-based generator."""
        super(Generator, self).__init__()

        c7s1_64 = ConvBlock(in_c, ngf, 7, stride=1, conv_padding=0, head_padding=3,
                            head_pad_type='reflect', norm_type=norm_type, act_type='relu')  # (N, 64, H, W)
        head_block = c7s1_64

        # Downsampling layers.
        d128 = ConvBlock(ngf, ngf * 2, 3, stride=2, conv_padding=1,
                         norm_type=norm_type, act_type='relu')  # (N, 128, H/2, W/2)
        d256 = ConvBlock(ngf * 2, ngf * 4, 3, stride=2, conv_padding=1,
                         norm_type=norm_type, act_type='relu')  # (N, 256, H/4, W/4)
        down_block = d128 + d256
        # Bockbone is resblock.
        resblocks = []
        for i in range(n_resblocks):
            resblocks += [ResBlock(ngf * 4, ngf * 4, 3, stride=1, conv_padding=1,
                                   norm_type=norm_type, act_type='relu')]  # (N, 256, H/4, W/4)
        # Upsampling layers.
        u128 = UpsampleBlock(ngf * 4, ngf * 2, upscale_factor=2, kernel_size=3, stride=2, conv_padding=1,
                             output_padding=1, up_mode=up_mode, norm_type=norm_type, act_type='relu')
        u64 = UpsampleBlock(ngf * 2, ngf, upscale_factor=2, kernel_size=3, stride=2, conv_padding=1,
                            output_padding=1, up_mode=up_mode, norm_type=norm_type, act_type='relu')
        up_block = u128 + u64
        # Tile layers.
        c7s1_3 = ConvBlock(ngf, out_c, 7, stride=1, conv_padding=0, head_padding=3,
                           head_pad_type='reflect', norm_type='none', act_type='none')
        tile_block = c7s1_3 + [nn.Tanh()]
        # Full architecture.
        arch = head_block + down_block + resblocks + up_block + tile_block
        # rename layers, every model must have unique names
        renamed_layers = []
        for idx, layer in enumerate(arch):
            renamed_layers.append((name + str(idx), layer))
        new_arch = OrderedDict(renamed_layers)
        self.generator = nn.Sequential(new_arch)

    def forward(self, input):
        """Forward process."""
        return self.generator(input)


def UpsampleBlock(in_nc, out_nc, upscale_factor=2, kernel_size=3, stride=1, conv_padding=1, output_padding=0,
                  up_mode='pixelshuffle', resize_mode='nearest', norm_type='none', act_type='relu'):
    """Upsample block.

    :param in_nc: the number of channels in input images
    :type in_nc: int
    :param out_nc: the number of channels in output images
    :type out_nc: int
    :param upscale_factor: factor of upsample
    :type upscale_factor: int
    :param kernel_size: kernel size of convolution layers
    :type kernel_size: int
    :param stride: stride of convolution layer
    :type stride: int
    :param conv_padding: padding size of conv layers
    :type conv_padding: int
    :param output_padding: padding size of deconv layers
    :type output_padding: int
    :param up_mode: type of upsample
    :type up_mode: str
    :param resize_mode: type of upsample
    :type resize_mode: str
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: activation layer
    :type act_type: str
    """
    # Normlization.
    if norm_type == 'none':
        norm = None
    elif norm_type == 'batch':
        norm = nn.BatchNorm2d(out_nc, affine=True)
    elif norm_type == 'instance':
        norm = nn.InstanceNorm2d(out_nc, affine=False)
    # Activation.
    if act_type == 'none':
        act = None
    elif act_type == 'relu':
        act = nn.ReLU(True)
    elif act_type == 'leakyrelu':
        act = nn.LeakyReLU(0.2, True)
    if up_mode == 'transpose':
        upsampler = nn.ConvTranspose2d(in_nc, out_nc, kernel_size=kernel_size,
                                       stride=stride, padding=conv_padding,
                                       output_padding=output_padding)
        arch = [upsampler, norm, act]
    elif up_mode == 'pixelshuffle':
        upsampler = nn.PixelShuffle(upscale_factor)
        conv = nn.Conv2d(in_nc, out_nc * (upscale_factor ** 2), kernel_size=kernel_size,
                         stride=stride, padding=conv_padding)
        arch = [conv, upsampler, norm, act]
    elif up_mode == 'resize':
        upsampler = nn.Upsample(scale_factor=upscale_factor, mode=resize_mode)
        conv = nn.Conv2d(in_nc, out_nc * (upscale_factor ** 2), kernel_size=kernel_size,
                         stride=stride, padding=conv_padding)
        arch = [upsampler, conv, norm, act]
    return [i for i in arch if i is not None]


class ResBlock(nn.Module):
    """Define a Resnet block.

    :param in_nc: the number of channels in input images
    :type in_nc: int
    :param out_nc: the number of channels in output images
    :type out_nc: int
    :param kernel_size: kernel size of convolution layers
    :type kernel_size: int
    :param stride: stride of convolution layer
    :type stride: int
    :param conv_padding: padding size of conv layers
    :type conv_padding: int
    :param padding_mode: padding mode of conv layers
    :type padding_mode: str
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: activation layer
    :type act_type: str
    """

    def __init__(self, in_nc, out_nc, kernel_size=3, stride=1, conv_padding=1, padding_mode='zeros',
                 norm_type='none', act_type='relu'):
        """Initialize the Resnet block."""
        super(ResBlock, self).__init__()
        conv_block0 = ConvBlock(in_nc, out_nc, kernel_size, stride=stride, conv_padding=conv_padding,
                                norm_type=norm_type, act_type=act_type)
        conv_block1 = ConvBlock(in_nc, out_nc, kernel_size, stride=stride, conv_padding=conv_padding,
                                norm_type=norm_type, act_type='none')
        conv_block = conv_block0 + conv_block1
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)."""
        out = x + self.conv_block(x)
        return out


def ConvBlock(in_nc, out_nc, kernel_size, stride=1, conv_padding=1, padding_mode='zeros',
              head_padding=0, head_pad_type='none', norm_type='none', act_type='relu'):
    """Conv layer with padding, normalization, activation.

    :param in_nc: the number of channels in input images
    :type in_nc: int
    :param out_nc: the number of channels in output images
    :type out_nc: int
    :param kernel_size: kernel size of convolution layers
    :type kernel_size: int
    :param stride: stride of convolution layer
    :type stride: int
    :param conv_padding: padding size of conv layers
    :type conv_padding: int
    :param padding_mode: padding mode of conv layers
    :type padding_mode: str
    :param head_padding: padding size of first conv layer
    :type head_padding: int
    :param head_pad_type: padding mode of first conv layer
    :type head_pad_type: str
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: activation layer
    :type act_type: str
    """
    # Padding type before convolution.
    if head_pad_type == 'none':
        head_pad = None
    elif head_pad_type == 'reflect':
        if vega.is_npu_device():
            # TO DO
            # Pad3d operator is not yet supported, temporarily circumvent it
            head_pad = ReflectionPad2d(head_padding)
        else:
            head_pad = nn.ReflectionPad2d(head_padding)
    elif head_pad_type == 'replicate':
        if vega.is_npu_device():
            head_pad = ReflectionPad2d(head_padding)
        else:
            head_pad = nn.ReflectionPad2d(head_padding)
    # Convolution.
    conv = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride,
                     padding=conv_padding, padding_mode=padding_mode)
    # Normlization.
    if norm_type == 'none':
        norm = None
    elif norm_type == 'batch':
        norm = nn.BatchNorm2d(out_nc, affine=True)
    elif norm_type == 'instance':
        norm = nn.InstanceNorm2d(out_nc, affine=False)
    # Activation.
    if act_type == 'none':
        act = None
    elif act_type == 'relu':
        act = nn.ReLU(True)
    elif act_type == 'leakyrelu':
        act = nn.LeakyReLU(0.2, True)
    # Merge all ConvBlocks.
    conv_block = [head_pad, conv, norm, act]
    return [i for i in conv_block if i is not None]


class PatchDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator.

    :param input_nc: the number of channels in input images
    :type input_nc: int
    :param ngf: the number of filters in the last conv layer
    :type ngf: int
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: activation layer
    :type act_type: str
    """

    def __init__(self, input_nc, ndf=64, norm_type='batch', act_type='leakyrelu', name='D_X'):
        """Construct a PatchGAN discriminator."""
        super(PatchDiscriminator, self).__init__()
        kernel_size = 4
        c64 = ConvBlock(input_nc, ndf, kernel_size, stride=2, conv_padding=1, padding_mode='zeros',
                        head_pad_type='none', norm_type='none', act_type=act_type)
        c128 = ConvBlock(ndf, ndf * 2, kernel_size, stride=2, conv_padding=1, padding_mode='zeros',
                         head_pad_type='none', norm_type='instance', act_type=act_type)
        c256 = ConvBlock(ndf * 2, ndf * 4, kernel_size, stride=2, conv_padding=1, padding_mode='zeros',
                         head_pad_type='none', norm_type='instance', act_type=act_type)
        c512 = ConvBlock(ndf * 4, ndf * 8, kernel_size, stride=1, conv_padding=1, padding_mode='zeros',
                         head_pad_type='none', norm_type='instance', act_type=act_type)
        c3 = ConvBlock(ndf * 8, ndf, kernel_size, stride=1, conv_padding=1, padding_mode='zeros',
                       head_pad_type='none', norm_type='none', act_type='none')
        # Full architecture.
        arch = c64 + c128 + c256 + c512 + c3
        # rename layers, every model must have unique names
        renamed_layers = []
        for idx, layer in enumerate(arch):
            renamed_layers.append((name + str(idx), layer))
        new_arch = OrderedDict(renamed_layers)
        self.discriminator = nn.Sequential(new_arch)

    def forward(self, input):
        """Forward process."""
        return self.discriminator(input)


class ReflectionPad2d(nn.ReflectionPad2d):
    """Defines ReflectionPad2d."""

    def forward(self, input):
        """Forward process."""
        dtype = input.dtype
        return F.pad(input.cpu().float(), self.padding, 'reflect').npu().to(dtype)
