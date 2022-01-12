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

from typing import Dict, Optional, Any
from modnas.registry.metrics import build
from modnas.registry import SPEC_TYPE
from .base import MetricsBase


def build_metrics_all(mt_configs: Optional[SPEC_TYPE], estim: Optional[Any] = None) -> Dict[str, MetricsBase]:
    """Build Metrics from configs."""
    metrics = {}
    MetricsBase.set_estim(estim)
    if mt_configs is None:
        mt_configs = {}
    if not isinstance(mt_configs, dict):
        mt_configs = {'default': mt_configs}
    for mt_name, mt_conf in mt_configs.items():
        mt = build(mt_conf)
        metrics[mt_name] = mt
    MetricsBase.set_estim(None)
    return metrics
