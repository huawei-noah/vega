# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import custom network."""

from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.networks.pytorch.customs", {
    "nago": ["network:NAGO"],
    "deepfm": ["network:DeepFactorizationMachineModel"],
    "autogate": ["network:AutoGateModel"],
    "autogroup": ["network:AutoGroupModel"],
    "bisenet": ["network:BiSeNet"],
    "modnas": ["network:ModNasArchSpace"],
    "mobilenetv2": ["network:MobileNetV2"],
    "gcn_regressor": ["network:GCNRegressor"],
})
