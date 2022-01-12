# -*- coding: utf-8 -*-

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
"""Manage LrScheduler class."""
import logging
from copy import deepcopy
import vega
from vega.common import ClassFactory, ClassType
from vega.common.config import Config
from ..config_bakcend_map import ConfigBackendMapping
from ..conf.lr_scheduler import LrSchedulerConfig, LrSchedulerMappingDict


class LrScheduler(object):
    """Register and call LrScheduler class."""

    config = LrSchedulerConfig()

    def __init__(self, config=None):
        """Initialize."""
        # register pytorch optim as default
        if config:
            self.config = Config(config)
            raw_config = deepcopy(self.config)
        else:
            self.config = LrScheduler.config
            raw_config = self.config.to_dict()
        raw_config.type = self.config.type
        map_dict = LrSchedulerMappingDict()
        self.map_config = ConfigBackendMapping(
            map_dict.type_mapping_dict, map_dict.params_mapping_dict).backend_mapping(raw_config)
        self._cls = ClassFactory.get_cls(ClassType.LR_SCHEDULER, self.map_config.type)

    def __call__(self, optimizer=None, epochs=None, steps=None):
        """Call lr scheduler class."""
        params = self.map_config.get("params", {})
        logging.debug("Call LrScheduler. name={}, params={}".format(self._cls.__name__, params))

        setattr(self._cls, "by_epoch", True)
        if hasattr(self.config, "by_epoch"):
            setattr(self._cls, "by_epoch", self.config.by_epoch)

        try:
            if params:
                return self._cls(optimizer, **params)
            else:
                return self._cls(optimizer)
        except Exception as ex:
            logging.error("Failed to call LrScheduler name={}, params={}".format(self._cls.__name__, params))
            raise ex


if vega.is_torch_backend():
    import torch.optim.lr_scheduler as torch_lr

    ClassFactory.register_from_package(torch_lr, ClassType.LR_SCHEDULER)
