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

"""Parameter Optimizer."""
import torch
from modnas.registry import parse_spec
from modnas.registry.optimizer import register, build


def get_optimizer(params, config, trainer_config=None):
    """Return a new Optimizer."""
    trainer_config = trainer_config or {}
    optim_type, optim_args = parse_spec(config)
    device_ids = trainer_config.get('device', [None])
    n_parallel = len(device_ids)
    if trainer_config.get('scale_lr', True) and 'lr' in optim_args:
        optim_args['lr'] *= n_parallel
    optimizer = build(optim_type, params, **optim_args)
    if n_parallel > 1:
        optimizer = torch.nn.DataParallel(optimizer, device_ids=device_ids).module
    return optimizer


module = torch.optim

for name, attr in module.__dict__.items():
    if name.startswith('__'):
        continue
    if not callable(attr):
        continue
    if name.islower():
        continue
    if name == 'Optimizer':
        continue
    register(attr)
