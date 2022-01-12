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

"""Defined Configs."""
from vega.common import ConfigSerializable


class AutoLaneConfig(ConfigSerializable):
    """AutoLaneConfig Config."""

    codec = 'AutoLaneNasCodec'
    random_ratio = 0.5
    num_mutate = 10
    max_sample = 100
    min_sample = 10
    objective_keys = "LaneMetric"

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AutoLaneConfig = {"codec": {"type": str},
                                "random_ratio": {"type": float},
                                "num_mutate": {"type": int},
                                "max_sample": {"type": int},
                                "min_sample": {"type": int}
                                }
        return rules_AutoLaneConfig
