# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Torch activation functions."""

import torch.nn as nn
from modnas.registry.arch_space import register

# torch activations
register(nn.ELU)
register(nn.Hardshrink)
register(nn.Hardtanh)
register(nn.LeakyReLU)
register(nn.LogSigmoid)
# register(torch.nn.MultiheadAttention)
register(nn.PReLU)
register(nn.ReLU)
register(nn.ReLU6)
register(nn.RReLU)
register(nn.SELU)
register(nn.CELU)
register(nn.Sigmoid)
register(nn.Softplus)
register(nn.Softshrink)
register(nn.Softsign)
register(nn.Tanh)
register(nn.Tanhshrink)
register(nn.Threshold)
