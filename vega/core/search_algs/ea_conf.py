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


class EAConfig(ConfigSerializable):
    """Base Config for EA."""

    num_individual = 8  # 128
    num_individual_per_iter = 1
    num_generation = 1
    start_ga_epoch = 0  # 50
    ga_interval = 1  # 10
    warmup = 0  # 50
