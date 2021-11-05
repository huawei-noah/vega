# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
