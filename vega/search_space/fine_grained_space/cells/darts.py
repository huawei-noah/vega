# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space import FineGrainedSpace, FineGrainedSpaceFactory
from vega.search_space.fine_grained_space.conditions import Cells, Collection
from vega.search_space.fine_grained_space.operators.darts import Cell, AuxiliaryHead
from vega.search_space.fine_grained_space.operators.torch_ops import op


@ClassFactory.register(ClassType.SEARCH_SPACE)
class Darts(FineGrainedSpace):
    """Darts Fine Grained Space."""

    def __init__(self, num_classes, stem, cells, ref, auxiliary, search, aux_size):
        super(Darts, self).__init__()
        self.stem = FineGrainedSpaceFactory.create_search_space(stem)
        self.cells = Collection(cells, ref)
        C_prev = self.cells.C_prev
        C_aux = self.cells.C_aux
        if not search and auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_aux, num_classes, aux_size)
        self.global_pooling = op.AdaptiveAvgPool2d(1)
        self.classifier = op.Linear(C_prev, self._classes)
