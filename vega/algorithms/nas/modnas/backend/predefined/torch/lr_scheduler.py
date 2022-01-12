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

"""LR Scheduler."""
import torch
from modnas.registry import parse_spec
from modnas.registry.lr_scheduler import register, build


def get_lr_scheduler(optimizer, config, trainer_config=None):
    """Return a new LR Scheduler."""
    trainer_config = trainer_config or {}
    lr_type, lr_args = parse_spec(config)
    if lr_type == 'CosineAnnealingLR':
        if 'T_max' not in lr_args and 'epochs' in trainer_config:
            lr_args['T_max'] = trainer_config['epochs']
    return build(lr_type, optimizer, **lr_args)


module = torch.optim.lr_scheduler

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
