# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Configs."""
from zeus.common import ConfigSerializable


class SpNasConfig(ConfigSerializable):
    """Sp NasConfig."""

    codec = 'SpNasCodec'
    total_list = "{local_base_path}/output/total_list_s.csv"
    sample_level = 'serial'
    max_sample = 10
    max_optimal = 5
    serial_settings = dict(num_mutate=3, addstage_ratio=0.05, expend_ratio=0.3, max_stages=6)
    regnition = False
    last_search_result = None
