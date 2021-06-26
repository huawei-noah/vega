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
