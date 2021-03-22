# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""ResNetVariant for Detection."""
from zeus.common import ClassType, ClassFactory
from .deformation import Deformation
from zeus.modules.operators import ops
from zeus.modules.operators import PruneConv2DFilter, PruneBatchNormFilter, PruneLinearFilter


@ClassFactory.register(ClassType.NETWORK)
class PruneDeformation(Deformation):
    """Prune any Network."""

    def __init__(self, desc, from_graph=False, weight_file=None):
        super(PruneDeformation, self).__init__(desc, from_graph, weight_file)
        self.is_adaptive_weight = True

    def deform(self):
        """Deform Network."""
        if not self.props:
            return
        for name, module in self.model.named_modules():
            if isinstance(module, ops.Conv2d):
                PruneConv2DFilter(module, self.props).filter()
            elif isinstance(module, ops.BatchNorm2d):
                PruneBatchNormFilter(module, self.props).filter()
            elif isinstance(module, ops.Linear):
                PruneLinearFilter(module, self.props).filter()
