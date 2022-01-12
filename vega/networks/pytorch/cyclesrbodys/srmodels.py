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

"""SR models."""
import math
import torch.nn as nn
from .networks import ConvBlock, UpsampleBlock, ShortcutBlock, ResBlock


class SRResNet(nn.Module):
    """SRResNet class.

    :param in_nc: input channel,3 for RGB and 1 for grayscale
    :type in_nc: int
    :param out_nc: output channel,3 for RGB and 1 for grayscale
    :type out_nc: int
    :param nf: number of filter in conv layers
    :type nf: int
    :param nb: number of residual blocks
    :type nb: int
    :param upscale: upscale number for SR
    :type upscale: int
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: type of activation layer
    :type act_type: str
    :param mode: permutation mode of norm, act, and conv
    :type mode: str
    :param res_scale: scale of residual
    :type residual: float
    :param upsample_mode: mode of upsample layer
    :type upsample_mode: str
    """

    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu',
                 up_mode='pixelshuffle'):
        """Initialize method."""
        super(SRResNet, self).__init__()

        fea_conv = ConvBlock(in_nc, nf, kernel_size=3, norm_type='none', act_type='none')
        resnet_blocks = []
        for _ in range(nb):
            resnet_blocks += ResBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type=act_type)
        resnet_blocks += ConvBlock(nf, nf, kernel_size=3, norm_type=norm_type, act_type='none')
        shortcut = ShortcutBlock(nn.Sequential(*resnet_blocks))
        HR_conv0 = ConvBlock(nf, nf, kernel_size=3, norm_type='none', act_type=act_type)
        HR_conv1 = ConvBlock(nf, out_nc, kernel_size=3, norm_type='none', act_type='none')
        n_upscale = int(math.log(upscale, 2))
        upsample_block = []
        for _ in range(n_upscale):
            upsample_block += UpsampleBlock(nf, nf, up_mode=up_mode, act_type=act_type)
        # Full architecture.
        arch = fea_conv + [shortcut] + upsample_block + HR_conv0 + HR_conv1

        self.model = nn.Sequential(*arch)

    def forward(self, x):
        """Forward process."""
        x = self.model(x)
        return x


class VDSR(nn.Module):
    """VDSR class.

    :param in_nc: input channel,3 for RGB and 1 for grayscale
    :type in_nc: int
    :param out_nc: output channel,3 for RGB and 1 for grayscale
    :type out_nc: int
    :param nf: number of filter in conv layers
    :type nf: int
    :param nb: number of residual blocks
    :type nb: int
    :param upscale: upscale number for SR
    :type upscale: int
    :param norm_type: type of normalization layer
    :type norm_type: str
    :param act_type: type of activation layer
    :type act_type: str
    :param mode: permutation mode of norm, act, and conv
    :type mode: str
    :param res_scale: scale of residual
    :type residual: float
    :param upsample_mode: mode of upsample layer
    :type upsample_mode: str
    """

    def __init__(self, in_nc, out_nc, nf, nb, norm_type='batch', upscale=4, act_type='relu',
                 up_mode='pixelshuffle'):
        super(VDSR, self).__init__()
        fea_conv = ConvBlock(in_nc, nf, kernel_size=3, stride=1, conv_padding=1, norm_type='none',
                             act_type='none')
        conv_blocks = []
        for _ in range(nb):
            conv_blocks += ConvBlock(nf, nf, kernel_size=3, stride=1, conv_padding=1,
                                     norm_type=norm_type, act_type=act_type)
        conv_blocks += ConvBlock(nf, nf, kernel_size=3, stride=1, conv_padding=1, norm_type=norm_type,
                                 act_type='none')
        shortcut = ShortcutBlock(nn.Sequential(*conv_blocks))
        HR_conv0 = ConvBlock(nf, nf, kernel_size=3, stride=1, conv_padding=1, norm_type='none', act_type=act_type)
        HR_conv1 = ConvBlock(nf, out_nc, kernel_size=3, stride=1, conv_padding=1, norm_type='none', act_type='none')

        n_upscale = int(math.log(upscale, 2))
        upsample_block = []
        for _ in range(n_upscale):
            upsample_block += UpsampleBlock(nf, nf, up_mode=up_mode, act_type=act_type)
        # Full architecture.
        arch = fea_conv + [shortcut] + upsample_block + HR_conv0 + HR_conv1
        self.model = nn.Sequential(*arch)

    def forward(self, x):
        """Forward process."""
        x = self.model(x)
        return x
