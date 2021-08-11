# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register metrics automatically."""

from .flops_and_params import calc_model_flops_params
from .forward_latency import calc_forward_latency, calc_forward_latency_on_host


def register_metrics(backend):
    """Import and register metrics automatically."""
    if backend == "pytorch":
        from . import pytorch
    elif backend == "tensorflow":
        from . import tensorflow
    elif backend == "mindspore":
        from . import mindspore
