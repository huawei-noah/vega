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


class AvazuCommonConfig(BaseConfig):
    """Default Dataset config for Avazu."""

    batch_size = 2000
    block_size = 2000000
    train_num_of_files = 17
    test_num_of_files = 5
    train_size = 32343173
    test_size = 8085794
    pos_train_samples = 0
    pos_test_samples = 0
    neg_train_samples = 0
    neg_test_samples = 0
    train_pos_ratio = 0
    test_pos_ratio = 0
    initialized = True
    max_length = 24
    num_of_feats = 645195
    feat_names = ['C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category',
                  'device_id', 'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16',
                  'C17', 'C18', 'C19', 'C20', 'C21', 'mday', 'hour', 'wday']
    feat_sizes = [7, 7, 3135, 3487, 24, 4002, 252, 28, 101449, 523672, 5925, 5, 4, 2417, 8, 9, 426, 4, 67, 166, 60, 10,
                  24, 7]
    random_sample = False
    shuffle_block = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Common_AutoLane = {"batch_size": {"type": int},
                                 "block_size": {"type": int},
                                 "train_num_of_files": {"type": int},
                                 "test_num_of_files": {"type": int},
                                 "train_size": {"type": int},
                                 "test_size": {"type": int},
                                 "pos_train_samples": {"type": int},
                                 "pos_test_samples": {"type": int},
                                 "neg_train_samples": {"type": int},
                                 "neg_test_samples": {"type": int},
                                 "train_pos_ratio": {"type": int},
                                 "test_pos_ratio": {"type": int},
                                 "initialized": {"type": bool},
                                 "max_length": {"type": int},
                                 "num_of_feats": {"type": int},
                                 "feat_names": {"type": list},
                                 "feat_sizes": {"type": list},
                                 "random_sample": {"type": bool},
                                 "shuffle_block": {"type": bool}
                                 }
        return rules_Common_AutoLane


class AvazuTrainConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuValConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuTestConfig(AvazuCommonConfig):
    """Default Dataset config for Avazu."""

    pass


class AvazuConfig(ConfigSerializable):
    """Default Dataset config for Avazu."""

    common = AvazuCommonConfig
    train = AvazuTrainConfig
    val = AvazuValConfig
    test = AvazuTestConfig

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_Avazu = {"common": {"type": dict},
                       "train": {"type": dict},
                       "val": {"type": dict},
                       "test": {"type": dict}
                       }
        return rules_Avazu

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {'common': cls.common,
                'train': cls.train,
                'val': cls.val,
                'test': cls.test
                }
