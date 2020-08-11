# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Manage LrScheduler class."""
import logging
import vega
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.config import obj2config
from ...conf import LrSchedulerConfig


class LrScheduler(object):
    """Register and call LrScheduler class."""

    config = LrSchedulerConfig()

    def __init__(self):
        """Initialize."""
        # register pytorch optim as default
        self._cls = ClassFactory.get_cls(ClassType.LR_SCHEDULER, self.config.type)

    def __call__(self, optimizer=None, epochs=None, steps=None):
        """Call lr scheduler class."""
        params = obj2config(self.config).get("params", {})
        logging.debug("Call LrScheduler. name={}, params={}".format(self._cls.__name__, params))

        if self._cls.__name__ == "CosineAnnealingLR":
            if params.get("T_max", -1) == -1:
                if params.get("by_epoch", True):
                    params["T_max"] = epochs
                else:
                    params["T_max"] = epochs * steps

        if self._cls.__name__ == "WarmupScheduler":
            params["epochs"] = epochs
            params["steps"] = steps

        try:
            if params and optimizer:
                return self._cls(optimizer, **params)
            elif optimizer:
                return self._cls(optimizer)
            else:
                return self._cls(**params)
        except Exception as ex:
            logging.error("Failed to call LrScheduler name={}, params={}".format(self._cls.__name__, params))
            raise ex


if vega.is_torch_backend():
    import torch.optim.lr_scheduler as torch_lr

    ClassFactory.register_from_package(torch_lr, ClassType.LR_SCHEDULER)
