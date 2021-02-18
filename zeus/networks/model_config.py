# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
from zeus.common import ConfigSerializable


class ModelConfig(ConfigSerializable):
    """Default Model config for Pipeline."""

    type = None
    model_desc = None
    model_desc_file = None
    pretrained_model_file = None
    models_folder = None
