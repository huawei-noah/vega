# -*- coding:utf-8 -*-

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
    "mnist": ["Mnist"],
    "sr_datasets": ["Set5", "Set14", "BSDS100"],
    "auto_lane_datasets": ["AutoLaneDataset"],
    "coco": ["CocoDataset", "DetectionDataset"],
    "glue": ["GlueDataset"],
    "spatiotemporal": ["SpatiotemporalDataset"],
    "reds": ["REDS"],
    "nasbench": ["Nasbench"],
    "pacs": ["Pacs"],
})
