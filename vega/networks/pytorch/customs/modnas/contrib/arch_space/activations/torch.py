# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch activation functions."""
import torch.nn
from modnas.registry.arch_space import register

modules = [
    'ELU',
    'Hardshrink',
    'Hardtanh',
    'LeakyReLU',
    'LogSigmoid',
    'PReLU',
    'ReLU',
    'ReLU6',
    'RReLU',
    'SELU',
    'CELU',
    'Sigmoid',
    'Softplus',
    'Softshrink',
    'Softsign',
    'Tanh',
    'Tanhshrink',
    'Threshold',
]

for name in modules:
    attr = getattr(torch.nn, name, None)
    if attr is not None:
        register(attr)
