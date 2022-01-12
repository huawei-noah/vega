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
                 net=None,
                 **kwargs):
        super().__init__()
        use_backend('torch')
        config = Config(kwargs)
        self.config = config
        self.net = None
        is_augment = True if config.get('proc') == 'augment' or config.get('arch_desc') is not None else False
        if not config.get('vega_no_construct', False) and is_augment:
            Config.apply(config, config.pop('augment', {}))
            self.net = get_default_constructors(self.config)(self.net)

    def forward(self, *args, **kwargs):
        """Compute forward output."""
        return self.net(*args, **kwargs)
