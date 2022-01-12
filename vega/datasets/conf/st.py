# -*- coding=utf-8 -*-

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
"""Default configs."""

from vega.common import ConfigSerializable
from .base import BaseConfig


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
