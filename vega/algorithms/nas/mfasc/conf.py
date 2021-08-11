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


class MFASCConfig(ConfigSerializable):
    """MF-ASC Config."""

    sample_size = 5000
    batch_size = 1000
    prior_rho = 1.0
    beta = 1.0
    max_budget = 1000
    hf_epochs = 20
    lf_epochs = 5
    fidelity_ratio = 2
    min_hf_sample_size = 3
    min_lf_sample_size = 10
    predictor_type = 'mfgpr'
    objective_keys = ["accuracy"]
