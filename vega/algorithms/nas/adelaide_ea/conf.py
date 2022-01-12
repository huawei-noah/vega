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


class AdelaideConfig(ConfigSerializable):
    """AdelaideMutate Config."""

    codec = 'AdelaideCodec'
    max_sample = 10
    random_file = "{local_base_path}/output/random/random.csv"
    objective_keys = ['IoUMetric', 'flops']

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_AdelaideConfig = {"codec": {"type": str},
                                "max_sample": {"type": int},
                                "random_file": {"type": str},
                                "objective_keys": {"type": (list, str)}
                                }
        return rules_AdelaideConfig
