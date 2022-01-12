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

"""Initialization constructor."""
import numpy as np
from modnas.registry.construct import register
from modnas.core.param_space import ParamSpace


@register
class DefaultInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, model):
        """Run constructor."""
        ParamSpace().reset()
        seed = self.seed
        if seed:
            np.random.seed(seed)
        return model
