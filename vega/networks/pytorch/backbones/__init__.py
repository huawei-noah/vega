# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import pytorch backbones."""

from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.networks.pytorch.backbones", {
    "getter": ["BackboneGetter", "ResNetBackbone"],
    "load_official_model": ["OffcialModelLoader"],
    "resnet_variant_det": ["ResNetVariantDet"],
    "resnext_variant_det": ["ResNeXtVariantDet"],
})
