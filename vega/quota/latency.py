# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Flops and Parameters Filter."""

import logging
import vega
from vega.metrics import calc_forward_latency_on_host
from vega.model_zoo import ModelZoo
from .quota_item_base import QuotaItemBase


class LatencyVerification(QuotaItemBase):
    """Latency Filter class."""

    def __init__(self, latency_range):
        self.latency_range = latency_range

    def verify_on_host(self, model_desc):
        """Filter function of latency."""
        model = ModelZoo.get_model(model_desc)
        count_input = self.get_input_data()
        trainer = vega.trainer(model_desc=model_desc)
        sess_config = trainer._init_session_config() if vega.is_tf_backend() else None
        latency = calc_forward_latency_on_host(model, count_input, sess_config)
        logging.info(f"Sampled model's latency: {latency}ms")
        if latency < self.latency_range[0] or latency > self.latency_range[1]:
            logging.info(f"The latency ({latency}) is out of range. Skip this network.")
            return False
        else:
            return True
