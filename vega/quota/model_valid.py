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

"""Model Valid Verification."""

import logging
import vega
from vega.model_zoo import ModelZoo
from .quota_item_base import QuotaItemBase


class ModelValidVerification(QuotaItemBase):
    """Model valid verification."""

    def verify(self, model_desc):
        """Filter function of latency."""
        try:
            model = ModelZoo.get_model(model_desc)
            count_input = self.get_input_data()
            if vega.is_ms_backend():
                from mindspore import context
                context.set_context(device_target="CPU")
            model(count_input)
            return True
        except Exception as e:
            logging.info(f"Invild model desc: {model_desc}, error: {e}")
            return False
