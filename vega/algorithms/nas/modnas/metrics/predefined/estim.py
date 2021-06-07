# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Estimator-based metrics."""
from ..base import MetricsBase
from modnas.registry.metrics import register


@register
class ValidateMetrics(MetricsBase):
    """Estimator validation metrics class."""

    def __init__(self, field=None):
        super().__init__()
        self.field = field

    def __call__(self, model):
        """Return metrics output."""
        estim = self.estim
        val_res = estim.valid_epoch(model=model)
        if isinstance(val_res, dict):
            field = self.field
            default_res = list(val_res.values())[0]
            if field is None:
                val_res = default_res
            elif field in val_res:
                val_res = val_res[field]
            else:
                self.logger.error('field \"{}\" not exists, using default'.format(field))
                val_res = default_res
        return val_res
