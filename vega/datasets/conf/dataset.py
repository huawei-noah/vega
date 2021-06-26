# -*- coding=utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Dataset configs."""

from vega.common import ConfigSerializable
from vega.common import ClassType


class DatasetConfig(ConfigSerializable):
    """Default Dataset config for Pipeline."""

    type = "Cifar10"
    _class_type = ClassType.DATASET
    _class_data = None

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules_dataset = {"type": {"type": str},
                         "common": {"type": dict},
                         "train": {"type": dict},
                         "test": {"type": dict}
                         }
        return rules_dataset
