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

"""Torch utils."""
import numpy as np
import torch
from modnas.utils import format_value, format_dict


_device = None


def version():
    """Return backend version information."""
    return format_dict({
        'torch': torch.__version__,
        'cuda': torch._C._cuda_getCompiledVersion(),
        'cudnn': torch.backends.cudnn.version(),
    }, sep=', ', kv_sep='=', fmt_key=False, fmt_val=False)


def init_device(device=None, seed=11235):
    """Initialize device and set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def set_device(device):
    """Set current device."""
    global _device
    _device = device


def get_device():
    """Return current device."""
    return _device


def get_dev_mem_used():
    """Return memory used in device."""
    return torch.cuda.memory_allocated() / 1024. / 1024.


def param_count(model, *args, format=True, **kwargs):
    """Return number of model parameters."""
    val = sum(p.data.nelement() for p in model.parameters())
    return format_value(val, *args, **kwargs) if format else val


def param_size(model, *args, **kwargs):
    """Return size of model parameters."""
    val = 4 * param_count(model, format=False)
    return format_value(val, *args, binary=True, **kwargs) if format else val


def model_summary(model):
    """Return model summary."""
    info = {
        'params': param_count(model, factor=2, prec=4),
        'size': param_size(model, factor=2, prec=4),
    }
    return 'Model summary: {}'.format(', '.join(['{k}={{{k}}}'.format(k=k) for k in info])).format(**info)


def clear_bn_running_statistics(model):
    """Clear BatchNorm running statistics."""
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()


def recompute_bn_running_statistics(model, trainer, num_batch=100, clear=True):
    """Recompute BatchNorm running statistics."""
    if clear:
        clear_bn_running_statistics(model)
    is_training = model.training
    model.train()
    with torch.no_grad():
        for _ in range(num_batch):
            try:
                trn_X, _ = trainer.get_next_train_batch()
            except StopIteration:
                break
            model(trn_X)
            del trn_X
    if not is_training:
        model.eval()
