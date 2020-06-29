# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import all torch operators."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from vega.core.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Esrn_Cat(nn.Module):
    """Call torch.cat."""

    def __init__(self):
        super(Esrn_Cat, self).__init__()

    def forward(self, x):
        """Forward x."""
        return torch.cat(list(x), 1)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class MicroDecoder_Upsample(nn.Module):
    """Call torch.Upsample."""

    def __init__(self, collect_inds, agg_concat):
        self.collect_inds = collect_inds
        self.agg_concat = agg_concat
        super(MicroDecoder_Upsample, self).__init__()

    def forward(self, x):
        """Forward x."""
        out = x[self.collect_inds[0]]
        for i in range(1, len(self.collect_inds)):
            collect = x[self.collect_inds[i]]
            if out.size()[2] > collect.size()[2]:
                # upsample collect
                collect = nn.Upsample(size=out.size()[2:], mode='bilinear', align_corners=True)(collect)
            elif collect.size()[2] > out.size()[2]:
                out = nn.Upsample(size=collect.size()[2:], mode='bilinear', align_corners=True)(out)
            if self.agg_concat:
                out = torch.cat([out, collect], 1)
            else:
                out += collect
        out = F.relu(out)
        return out
