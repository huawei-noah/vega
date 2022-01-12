# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
