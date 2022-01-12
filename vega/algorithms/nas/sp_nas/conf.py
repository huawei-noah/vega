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


class SpNasConfig(ConfigSerializable):
    """Sp NasConfig."""

    max_sample = 20
    max_optimal = 5
    num_mutate = 3
    objective_keys = ['mAP', 'params']
    add_stage_ratio = 0.05
    expend_ratio = 0.3
    max_stages = 6

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_SpNasConfig = {
            "max_sample": {"type": int},
            "max_optimal": {"type": int},
            "num_mutate": {"type": int},
            "max_stages": {"type": int},
        }
        return rules_SpNasConfig
