# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is the class for SR dataset."""
from vega.core.common.class_factory import ClassFactory, ClassType
from .div2k import DIV2K


@ClassFactory.register(ClassType.DATASET)
class Set5(DIV2K):
    """Set5 dataset, its class interface is same as DIV2K."""

    pass


@ClassFactory.register(ClassType.DATASET)
class Set14(DIV2K):
    """Set14 dataset, its class interface is same as DIV2K."""

    pass


@ClassFactory.register(ClassType.DATASET)
class BSDS100(DIV2K):
    """BSDS100 dataset, its class interface is same as DIV2K."""

    pass
