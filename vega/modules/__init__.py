# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register modules automatically."""

from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.modules", {
    "module": ["network:Module"],
})


def register_modules():
    """Import and register modules automatically."""
    from . import blocks
    from . import cells
    from . import connections
    from . import operators
    from . import preprocess
    from . import loss
    from . import getters
    from . import necks
    from . import backbones
    from . import distillation
