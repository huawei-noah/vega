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

"""Flops and Parameters Filter."""

import logging
from vega.metrics import calc_model_flops_params
from vega.model_zoo import ModelZoo
from .quota_item_base import QuotaItemBase

logger = logging.getLogger(__name__)


class FlopsParamsVerification(QuotaItemBase):
    """Flops and Parameters Filter class."""

    def __init__(self, params_range, flops_range):
        self.params_range = params_range
        self.flops_range = flops_range

    def verify(self, model_desc=None):
        """Verify params and flops."""
        try:
            model = ModelZoo.get_model(model_desc)
            count_input = self.get_input_data()
            flops, params = calc_model_flops_params(model, count_input)
            flops, params = flops * 1e-9, params * 1e-3
            result = flops > self.flops_range[0] and flops < self.flops_range[1]
            result = result and params > self.params_range[0] and params < self.params_range[1]
            if not result:
                logger.info(f"params ({params}) or flops ({flops}) out of range.")
            return result
        except Exception as e:
            logging.info(f"Invild model desc: {model_desc}, error: {e}")
            return False
