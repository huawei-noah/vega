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

"""Uniformly samples and trains subnets."""
import random
from modnas.estim.predefined.default import DefaultEstim
from modnas.registry.estim import register
from modnas.core.param_space import ParamSpace


@register
class RandomSamplingEstim(DefaultEstim):
    """Trains a subnet uniformly sampled from the supernet in each step."""

    def __init__(self, *args, seed=1, save_best=True, **kwargs):
        super().__init__(*args, save_best=save_best, **kwargs)
        random.seed(seed)
        self.space_size = ParamSpace().categorical_size()

    def loss(self, data, output=None, model=None, mode=None):
        """Sample a subnet and compute its loss."""
        loss = super().loss(data, output, model, mode)
        params = ParamSpace().get_categorical_params(random.randint(0, self.space_size - 1))
        ParamSpace().update_params(params)
        return loss
