# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined FMD Unit."""
import torch
import torch.nn.functional as F
import numpy as np
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.OPS)
class FMDUnit(Network):
    """Basic class for feature map distortion.

    :param drop_prob: probability of an element to be dropped.
    :type drop_prob: float
    :param block_size: size of the block to drop.
    :type block_size: int
    """

    def __init__(self, drop_prob, block_size, args=None):
        """Init FMDUnit."""
        super(FMDUnit, self).__init__()
        self.drop_prob = drop_prob
        self.weight_behind = None
        self.weight_record = None
        self.alpha = args.alpha
        self.block_size = block_size

    def forward(self, x):
        """Forward."""
        if not self.training:
            return x
        else:
            width = x.size(3)
            seed_drop_rate = self.drop_prob * width**2 / self.block_size**2 / (width - self.block_size + 1)**2
            valid_block_center = torch.zeros(width, width, device=x.device).float()
            valid_block_center[int(self.block_size // 2):(width - (self.block_size - 1) // 2),
                               int(self.block_size // 2):(width - (self.block_size - 1) // 2)] = 1.0
            valid_block_center = valid_block_center.unsqueeze(0).unsqueeze(0)
            randnoise = torch.rand(x.shape, device=x.device)
            block_pattern = ((1 - valid_block_center + float(1 - seed_drop_rate) + randnoise) >= 1).float()
            if self.block_size == width:
                block_pattern = torch.min(block_pattern.view(x.size(0), x.size(1),
                                          x.size(2) * x.size(3)), dim=2)[0].unsqueeze(-1).unsqueeze(-1)
            else:
                block_pattern = -F.max_pool2d(input=-block_pattern, kernel_size=(self.block_size, self.block_size),
                                              stride=(1, 1), padding=self.block_size // 2)
            if self.block_size % 2 == 0:
                block_pattern = block_pattern[:, :, :-1, :-1]
            percent_ones = block_pattern.sum() / float(block_pattern.numel())
            if not (self.weight_behind is None) and not(len(self.weight_behind) == 0):
                wtsize = self.weight_behind.size(3)
                weight_max = self.weight_behind.max(dim=0, keepdim=True)[0]
                sig = torch.ones(weight_max.size(), device=weight_max.device)
                sig[torch.rand(weight_max.size(), device=sig.device) < 0.5] = -1
                weight_max = weight_max * sig
                weight_mean = weight_max.mean(dim=(2, 3), keepdim=True)
                if wtsize == 1:
                    weight_mean = 0.1 * weight_mean
                self.weight_record = weight_mean
            var = torch.var(x).clone().detach()
            if not (self.weight_behind is None) and not(len(self.weight_behind) == 0):
                noise = self.alpha * weight_mean * (var**0.5) * torch.randn(*x.shape, device=x.device)
            else:
                noise = self.alpha * 0.01 * (var**0.5) * torch.randn(*x.shape, device=x.device)
            x = x * block_pattern
            noise = noise * (1 - block_pattern)
            x = x + noise
            x = x / percent_ones
            return x


@NetworkFactory.register(NetTypes.OPS)
class LinearScheduler(Network):
    """LinearScheduler class.

    :param dropblock: drop block.
    :type dropblock: nn.Module
    :param start_value: drop rate start value.
    :type start_value: float
    :param stop_value: drop rate stop value.
    :type stop_value: float
    :param nr_steps: drop rate decay steps.
    :type nr_steps: int
    """

    def __init__(self, fmdblock, start_value, stop_value, nr_steps):
        super(LinearScheduler, self).__init__()
        self.fmdblock = fmdblock
        self.i = 0
        self.dis_values = np.linspace(start=int(start_value), stop=int(stop_value), num=int(nr_steps))

    def forward(self, x):
        """Forward."""
        return self.fmdblock(x)

    def step(self):
        """Step."""
        if self.i < len(self.dis_values):
            self.fmdblock.drop_prob = self.dis_values[self.i]
        self.i += 1
