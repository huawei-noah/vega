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

"""This is SearchSpace for head."""
from vega.modules.module import Module
from vega.common import ClassFactory, ClassType
from vega.modules.operators import ops


@ClassFactory.register(ClassType.NETWORK)
class LinearClassificationHead(Module):
    """Create LinearClassificationHead SearchSpace."""

    def __init__(self, base_channel, num_classes):
        """Create layers.

        :param base_channel: base_channel
        :type base_channel: int
        :param num_class: number of class
        :type num_class: int
        """
        super(LinearClassificationHead, self).__init__()
        self.avgpool = ops.AdaptiveAvgPool2d(output_size=(1, 1))
        self.view = ops.View()
        self.linear = ops.Linear(in_features=base_channel, out_features=num_classes)


@ClassFactory.register(ClassType.NETWORK)
class AuxiliaryHead(Module):
    """Auxiliary Head of Network.

    :param C: input channels
    :type C: int
    :param num_classes: numbers of classes
    :type num_classes: int
    :param input_size: input size
    :type input_size: int
    """

    def __init__(self, C, num_classes, input_size):
        """Init AuxiliaryHead."""
        super(AuxiliaryHead, self).__init__()
        stride = input_size - 5
        self.relu1 = ops.Relu(inplace=True)
        self.avgpool1 = ops.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False)
        self.conv1 = ops.Conv2d(C, 128, 1, bias=False)
        self.batchnorm1 = ops.BatchNorm2d(128)
        self.relu2 = ops.Relu(inplace=True)
        self.conv2 = ops.Conv2d(128, 768, 2, bias=False)
        self.batchnorm2 = ops.BatchNorm2d(768)
        self.relu3 = ops.Relu(inplace=True)
        self.view = ops.View()
        self.classifier = ops.Linear(768, num_classes)
