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


class PBAPolicyConfig(ConfigSerializable):
    """PBA Policy Config."""

    total_epochs = 81
    config_count = 16
    each_epochs = 3
    total_rungs = 200


class PBAConfig(ConfigSerializable):
    """PBA Config."""

    policy = PBAPolicyConfig
    objective_keys = 'accuracy'
    transformers = dict(Cutout=True,
                        Rotate=True,
                        Translate_X=True,
                        Translate_Y=True,
                        Brightness=True,
                        Color=True,
                        Invert=True,
                        Sharpness=True,
                        Posterize=True,
                        Shear_X=True,
                        Solarize=True,
                        Shear_Y=True,
                        Equalize=True,
                        AutoContrast=True,
                        Contras=True)
