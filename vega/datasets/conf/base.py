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


class BaseConfig(ConfigSerializable):
    """Base config of dataset."""

    data_path = None
    batch_size = 1
    num_workers = 0
    imgs_per_gpu = 1,
    shuffle = False
    distributed = False
    download = False
    pin_memory = True
    drop_last = True
    transforms = []
    buffer_size = 128

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Base = {"data_path": {"type": (str, None)},
                      "batch_size": {"type": int},
                      "num_workers": {"type": int},
                      "shuffle": {"type": bool},
                      "distributed": {"type": bool},
                      "download": {"type": bool},
                      "pin_memory": {"type": bool},
                      "drop_last": {"type": bool},
                      "transforms": {"type": list},
                      }
        return rules_Base


class ExtTrainConfig(BaseConfig):
    """Extension config."""

    pass


class ExtValConfig(BaseConfig):
    """Extension config."""

    pass


class ExtTestConfig(BaseConfig):
    """Extension config."""

    pass


class ExtDatasetConfig(ConfigSerializable):
    """Extension dataset config."""

    common = BaseConfig
    train = ExtTrainConfig
    val = ExtValConfig
    test = ExtTestConfig
