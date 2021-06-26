# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Default configs."""

from .base import BaseConfig
from vega.common import ConfigSerializable


class SpatiotemporalConfig(BaseConfig):
    """Default Dataset config for SpatiotemporalConfig."""

    n_his = 12
    n_pred = 4
    batch_size = 32
    test_portion = 0.2
    train_portion = 0.9
    is_spatiotemporal = True

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Base = {"data_path": {"type": (str)},
                      "n_his": {"type": int},
                      "n_pred": {"type": bool},
                      "train_portion": {"type": float},
                      "is_spatiotemporal": {"type": bool},
                      }
        return rules_Base


class SpatiotemporalDatasetConfig(ConfigSerializable):
    """Dummy dataset config."""

    common = SpatiotemporalConfig
    train = SpatiotemporalConfig
    val = SpatiotemporalConfig
    test = SpatiotemporalConfig
