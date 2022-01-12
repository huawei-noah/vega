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

"""Estimator-based metrics."""
from modnas.registry.metrics import register
from ..base import MetricsBase


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
