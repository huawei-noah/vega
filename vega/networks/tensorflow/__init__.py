# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Lazy import tensorflow networks."""

from .network import Sequential
from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.networks.tensorflow", {
    "resnet_tf": ["ResNetTF", 'ResNetSlim'],
    # backbones
    "backbones.resnet_det": ["ResNetDet"],
    # customs
    "customs.edvr.edvr": ["EDVR"],
    "customs.gcn_regressor": ["GCNRegressor"],
    # detectors
    "detectors.faster_rcnn_trainer_callback": ["FasterRCNNTrainerCallback"],
    "detectors.faster_rcnn": ["FasterRCNN"],
    "detectors.tf_optimizer": ["TFOptimizer"],
    # losses
    "losses.cross_entropy_loss": ["CrossEntropyLoss"],
    "losses.mix_auxiliary_loss": ["MixAuxiliaryLoss"],
    "losses.charbonnier": ["CharbonnierLoss"],
    # necks
    "necks.mask_rcnn_box": ["MaskRCNNBox"],
})

ClassFactory.lazy_register("vega.networks.tensorflow.utils", {
    "anchor_utils.anchor_generator": ["AnchorGenerator"],
    "hyperparams.initializer": ["Initializer"],
    "hyperparams.regularizer": ["Regularizer"],
})
