# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ModularNAS arch space wrapper."""

try:
    from torch.nn import Module
except ModuleNotFoundError:
    Module = object
from vega.common import ClassFactory, ClassType
from modnas.backend import use as use_backend
from modnas.utils.config import Config
from modnas.utils.wrapper import get_default_constructors
import modnas.utils.predefined


@ClassFactory.register(ClassType.NETWORK)
class ModNasArchSpace(Module):
    """ModularNAS Architecture Space."""

    def __init__(self,
                 construct=None,
                 desc_construct=None,
                 net=None,
                 **kwargs):
        super().__init__()
        use_backend('torch')
        config = Config(kwargs)
        fully_train = config.get('arch_desc', config.get('fully_train')) or False
        config['construct'] = (desc_construct if fully_train is not False else construct) or {}
        self.config = config
        self.constructor = None
        self.net = None
        if fully_train is not False:
            self.build_net()

    def build_net(self):
        """Build network with constructors."""
        self.constructor = get_default_constructors(self.config)
        self.net = self.constructor(self.net)

    def forward(self, *args, **kwargs):
        """Compute forward output."""
        return self.net(*args, **kwargs)
