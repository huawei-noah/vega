# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from modnas.registry.metrics import build
from modnas.registry import SPEC_TYPE
from .base import MetricsBase
from typing import Dict, Optional, Any


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
