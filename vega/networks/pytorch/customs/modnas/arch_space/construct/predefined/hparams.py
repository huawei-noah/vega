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

"""Hyperparameter constructor."""
from typing import Dict, List, Union
from modnas.registry.construct import register
from modnas.core.params import Numeric, Categorical


@register
class DefaultHParamSpaceConstructor():
    """Constructor that generates parameters from config."""

    def __init__(self, params: Union[Dict, List]) -> None:
        if isinstance(params, dict):
            self.params = params.items()
        elif isinstance(params, list):
            self.params = [(None, p) for p in params]

    def __call__(self, model: None) -> None:
        """Run constructor."""
        del model
        for k, v in self.params:
            if isinstance(v, list) and len(v) == 1 and isinstance(v[0], list):
                _ = Numeric(low=v[0][0], high=v[0][1], name=k)
            elif isinstance(v, list):
                _ = Categorical(choices=v, name=k)
            else:
                raise ValueError('support categorical and numeric hparams only')
