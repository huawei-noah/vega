# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import dataset."""

from zeus.common.class_factory import ClassFactory


ClassFactory.lazy_register("zeus.datasets.common", {
    "avazu": ["AvazuDataset"],
    "cifar10": ["Cifar10"],
    "cifar100": ["Cifar100"],
    "div2k": ["DIV2K"],
    "cls_ds": ["ClassificationDataset"],
    "cityscapes": ["Cityscapes"],
    "div2k_unpair": ["Div2kUnpair"],
    "fmnist": ["FashionMnist"],
    # "imagenet": ["Imagenet"],
    "mnist": ["Mnist"],
    "sr_datasets": ["Set5", "Set14", "BSDS100"],
    "auto_lane_datasets": ["AutoLaneDataset"],
    "coco": ["CocoDataset"],
    "mrpc": ["MrpcDataset"],
    "spatiotemporal": ["SpatiotemporalDataset"]
})
