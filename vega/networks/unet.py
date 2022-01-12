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
from vega.common import ClassFactory, ClassType
from vega.modules.module import Module
from vega.modules.connections import Sequential
from vega.modules.operators.ops import Conv2d, BatchNorm2d, Relu, MaxPool2d, ConvTranspose2d, concat, Sigmoid


@ClassFactory.register(ClassType.NETWORK)
class Unet(Module):
    """Create ResNet SearchSpace."""

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        """Create layers.

        :param in_channels: in channel
        :type in_channels: int
        :param out_channels: out_channels
        :type out_channels: int
        :param init_features: features
        :type init_features: int
        """
        super(Unet, self).__init__()
        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 8, features * 16)

        self.upconv4 = ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8)
        self.upconv3 = ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4)
        self.upconv2 = ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.conv = Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def _block(self, in_channels, features):
        """Make block."""
        conv1 = Conv2d(in_channels=in_channels, out_channels=features, padding=1)
        bn1 = BatchNorm2d(num_features=features)
        relu1 = Relu(inplace=True)
        conv2 = Conv2d(in_channels=features, out_channels=features, padding=1)
        bn2 = BatchNorm2d(num_features=features)
        relu2 = Relu(inplace=True)
        block = [conv1, bn1, relu1, conv2, bn2, relu2]
        backbone = Sequential()
        for block_site in block:
            backbone.append(block_site)
        return backbone

    def call(self, x):
        """Forward function.

        :return: output of block
        :rtype: tensor
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = concat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = concat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = concat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = concat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return Sigmoid(self.conv(dec1))
