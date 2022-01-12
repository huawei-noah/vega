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

"""Quota."""

import logging
from vega.common.class_factory import ClassFactory, ClassType
from vega.common.general import General
from .model_valid import ModelValidVerification
from .flops_params import FlopsParamsVerification
from .quota_affinity import QuotaAffinity
from .latency import LatencyVerification

logger = logging.getLogger(__name__)


@ClassFactory.register(ClassType.QUOTA)
class Quota(object):
    """Determine whether to terminate duration."""

    def __init__(self, config=None):
        self.enable = False
        self.params_range = []
        self.flops_range = []
        self.host_latency_range = []
        self.device_latency_range = []
        self.model_valid_enable = False
        self.affinity_enable = False
        self.runtime = None
        self._set_config(config or General.quota)

    def _set_config(self, config):
        """Set quota.

        config examples:
            "accuray > 12 and flops in [23, 45] and model_valid"
            "accuray in [12, 14]"
            "accuray > 12"
            "model_valid"
        """
        if config is None or config.strip() == "":
            return
        items = config.split("and")
        for item in items:
            if "model_valid" in item:
                self.model_valid_enable = True
            elif "affinity" in item:
                self.affinity_enable = True
            elif "params" in item:
                self.params_range = self._set_value(item)
            elif "flops" in item:
                self.flops_range = self._set_value(item)
            elif "host_latency" in item:
                self.host_latency_range = self._set_value(item)
            elif "device_latency" in item:
                self.device_latency_range = self._set_value(item)
            elif "runtime" in item:
                self.runtime = float(item.split("<")[1].strip())
        self.enable = True

    def _set_value(self, value):
        if "in" in value:
            value_range = value.split("in")[1].strip()[1:-1].split(",")
            return [float(value_range[0]), float(value_range[1])]
        elif ">" in value:
            return [float(value.split(">")[1].strip()), float("inf")]
        elif "<" in value:
            return [-float('inf'), float(value.split("<")[1].strip())]
        else:
            raise ValueError(f"valid quota value: {value}")

    def verify_sample(self, model_desc):
        """Verify model_valid, flops, params."""
        if not self.enable:
            return True
        if self.model_valid_enable and len(self.flops_range) == 0 and len(self.params_range) == 0:
            result = ModelValidVerification().verify(model_desc)
            if not result:
                return False
        if len(self.flops_range) == 2 or len(self.params_range) == 2:
            result = FlopsParamsVerification(self.params_range, self.flops_range).verify(model_desc)
            if not result:
                return False
        if len(self.host_latency_range) == 2:
            result = LatencyVerification(self.host_latency_range).verify_on_host(model_desc)
            if not result:
                return False
        return True

    def verify_affinity(self, model_desc):
        """Verify affinity."""
        if not self.enable or not self.affinity_enable:
            return True
        affinity = QuotaAffinity(General.affinity_config)
        return affinity.is_affinity(model_desc)

    def adjuest_pipeline_by_runtime(self, user_config):
        """Adjuest pipeline by runtime."""
        if not self.enable or self.runtime is None:
            return True
        return True

    def verify_metric(self, model_desc):
        """Verify metrics."""
        return True

    @property
    def quota_reached(self):
        """Return True if reach the limits."""
        return False
