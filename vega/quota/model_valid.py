# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model Valid Verification."""

import logging
from .quota_item_base import QuotaItemBase
from vega.model_zoo import ModelZoo


class ModelValidVerification(QuotaItemBase):
    """Model valid verification."""

    def verify(self, model_desc):
        """Filter function of latency."""
        try:
            model = ModelZoo.get_model(model_desc)
            count_input = self.get_input_data()
            model(count_input)
            return True
        except Exception as e:
            logging.info(f"Invild model desc: {model_desc}, error: {e}")
            return False
