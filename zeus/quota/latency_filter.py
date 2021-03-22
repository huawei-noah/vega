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
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.metrics import calc_forward_latency
from .filter_terminate_base import FilterTerminateBase


@ClassFactory.register(ClassType.QUOTA)
class LatencyFilter(FilterTerminateBase):
    """Latency Filter class."""

    def __init__(self):
        super(LatencyFilter, self).__init__()
        self.max_latency = self.restrict_config.latency
        if self.max_latency is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            self.dataset = dataset_cls()
            from zeus.datasets import Adapter
            self.dataloader = Adapter(self.dataset).loader

    def is_filtered(self, desc=None):
        """Filter function of latency."""
        if self.max_latency is None:
            return False
        model, count_input = self.get_model_input(desc)
        trainer = ClassFactory.get_cls(ClassType.TRAINER)(model_desc=desc)
        sess_config = trainer._init_session_config() if zeus.is_tf_backend() else None
        latency = calc_forward_latency(model, count_input, sess_config)
        logging.info('Sampled model\'s latency: {}ms'.format(latency))
        if latency > self.max_latency:
            logging.info('The latency is out of range. Skip this network.')
            return True
        else:
            return False
