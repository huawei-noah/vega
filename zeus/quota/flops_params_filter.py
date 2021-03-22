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
from zeus.common import ClassFactory, ClassType
from zeus.metrics import calc_model_flops_params
from .filter_terminate_base import FilterTerminateBase

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.QUOTA)
class FlopsParamsFilter(FilterTerminateBase):
    """Flops and Parameters Filter class."""

    def __init__(self):
        super(FlopsParamsFilter, self).__init__()
        self.flops_range = self.restrict_config.flops
        self.params_range = self.restrict_config.params
        if self.flops_range and not isinstance(self.flops_range, list):
            self.flops_range = [0., self.flops_range]
        if self.params_range and not isinstance(self.params_range, list):
            self.params_range = [0., self.params_range]
        if self.flops_range is not None or self.params_range is not None:
            dataset_cls = ClassFactory.get_cls(ClassType.DATASET)
            self.dataset = dataset_cls()
            from zeus.datasets import Adapter
            self.dataloader = Adapter(self.dataset).loader

    def is_filtered(self, desc=None):
        """Filter function of Flops and Params."""
        if self.flops_range is None and self.params_range is None:
            return False
        model, count_input = self.get_model_input(desc)
        flops, params = calc_model_flops_params(model, count_input)
        flops, params = flops * 1e-9, params * 1e-3
        if self.flops_range is not None:
            if flops < self.flops_range[0] or flops > self.flops_range[1]:
                logger.info("The flops {} is out of range. Skip this network.".format(flops))
                return True
        if self.params_range is not None:
            if params < self.params_range[0] or params > self.params_range[1]:
                logger.info("The parameters {} is out of range. Skip this network.".format(params))
                return True
        return False
