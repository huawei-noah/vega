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
"""Layer Creator."""
import torch.nn as nn


class Meta(type):
    """Meta class of LayerCreator."""

    def __new__(cls, name, bases, namespace, **kwargs):
        """Construct class."""
        return super().__new__(cls, name, bases, namespace, **kwargs)

    def __init__(cls, name, bases, namespace, **kwargs):
        """Initialize class."""
        super().__init__(name, bases, namespace, **kwargs)
        if not hasattr(cls, 'registory'):
            cls.registory = {}
        else:
            cls.registory[name] = cls


class LayerCreator(object, metaclass=Meta):
    """This is a base class of all layers.

    :param type: type of layer
    :type type: str
    """

    def __new__(cls, type, **kwargs):
        """Construct class."""
        child = super().__new__(cls.registory[type])
        return child

    def __init__(self, *args, **kwargs):
        """Initialize class."""
        pass

    def get_name(self, magic_number=None):
        """Get the name of target layer."""
        if magic_number is not None:
            return f'{self.__class__.__name__.lower()}{magic_number}'
        else:
            return f'{self.__class__.__name__.lower()}'

    def create_layer(self, *args, **kwargs):
        """Create target layer."""
        raise NotImplementedError('LayerCreator is not a true type layer.')


class BN(LayerCreator):
    """This is the class of BN layer."""

    def __init__(self, type, requires_grad):
        super().__init__()
        self.requires_grad = requires_grad

    def create_layer(self, num_features):
        """Create target layer."""
        layer = nn.BatchNorm2d(num_features=num_features)
        for param in layer.parameters():
            param.requires_grad = self.requires_grad
        return layer


class Conv(LayerCreator):
    """This is the class of Conv layer."""

    def __init__(self, type, *args, **kwargs):
        """Initialize class."""
        super().__init__()

    def create_layer(self, *args, **kwargs):
        """Create target layer."""
        layer = nn.Conv2d(*args, **kwargs)
        return layer


class ReLU(LayerCreator):
    """This is the class of ReLU layer."""

    def __init__(self, type, *args, **kwargs):
        """Initialize class."""
        super().__init__()

    def create_layer(self, *args, **kwargs):
        """Create target layer."""
        layer = nn.ReLU(*args, **kwargs)
        return layer
