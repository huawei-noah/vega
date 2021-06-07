# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch utils."""
import numpy as np
import torch
from modnas.utils import format_value


def init_device(device=None, seed=11235):
    """Initialize device and set seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


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
