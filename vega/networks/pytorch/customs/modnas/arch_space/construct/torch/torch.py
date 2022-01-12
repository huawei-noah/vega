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

"""Torch constructors."""
import torch
from modnas.registry.construct import register
from modnas.arch_space.slot import Slot
from modnas.arch_space import ops
from modnas.core.param_space import ParamSpace
from modnas.utils.logging import get_logger
from modnas import backend


logger = get_logger('construct')


def parse_device(device):
    """Return device ids from config."""
    if isinstance(device, int):
        device = str(device)
    if not isinstance(device, str):
        return []
    device = device.lower()
    if device in ['cpu', 'nil', 'none']:
        return []
    if device == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in device.split(',')]


def configure_ops(new_config):
    """Set global operator config."""
    config = ops.config
    config.update(new_config)
    if isinstance(config.ops_order, str):
        config.ops_order = config.ops_order.split('_')
    if config.ops_order[-1] == 'bn':
        config.conv.bias = False
    if config.ops_order[0] == 'act':
        config.act.inplace = False
    logger.info('ops config: {}'.format(config.to_dict()))


@register
class TorchInitConstructor():
    """Constructor that initializes the architecture space."""

    def __init__(self, seed=None, device=None, ops_conf=None):
        self.seed = seed
        self.device = device
        self.ops_conf = ops_conf

    def __call__(self, model):
        """Run constructor."""
        Slot.reset()
        ParamSpace().reset()
        seed = self.seed
        if seed:
            backend.init_device(self.device, seed)
        configure_ops(self.ops_conf or {})
        return model


@register
class TorchToDevice():
    """Constructor that moves model to some device."""

    def __init__(self, device='all', data_parallel=True):
        device_ids = parse_device(device) or [None]
        self.device_ids = device_ids
        self.data_parallel = data_parallel

    def __call__(self, model):
        """Run constructor."""
        if model is None:
            return
        device_ids = self.device_ids
        backend.set_device(device_ids[0])
        if device_ids[0] is not None:
            torch.cuda.set_device(device_ids[0])
        model.to(device=device_ids[0])
        if self.data_parallel and len(device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        return model


@register
class TorchCheckpointLoader():
    """Constructor that loads model checkpoints."""

    def __init__(self, path):
        logger.info('Loading torch checkpoint from {}'.format(path))
        self.chkpt = torch.load(path)

    def __call__(self, model):
        """Run constructor."""
        model.load_state_dict(self.chkpt)
        return model
