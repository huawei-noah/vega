# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import dataset."""

from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.datasets.common", {
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
    "coco": ["CocoDataset", "DetectionDataset"],
    "glue": ["GlueDataset"],
    "spatiotemporal": ["SpatiotemporalDataset"],
    "reds": ["REDS"],
    "nasbench": ["Nasbench"],
})
