# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register network automatically."""

from vega.common.class_factory import ClassFactory
from .network_desc import NetworkDesc


ClassFactory.lazy_register("vega.networks", {
    "adelaide": ["AdelaideFastNAS"],
    "bert": ["BertClassification", "TinyBertForPreTraining", "BertClassificationHeader"],
    "dnet": ["DNet", "DNetBackbone"],
    "erdb_esr": ["ESRN"],
    "faster_backbone": ["FasterBackbone"],
    "faster_rcnn": ["FasterRCNN"],
    "mobilenet": ["MobileNetV3Tiny", "MobileNetV2Tiny"],
    "mobilenetv3": ["MobileNetV3Small", "MobileNetV3Large"],
    "necks": ["FPN"],
    "quant": ["Quantizer"],
    "resnet_det": ["ResNetDet"],
    "resnet_general": ["ResNetGeneral"],
    "resnet": ["ResNet"],
    "resnext_det": ["ResNeXtDet"],
    "sgas_network": ["SGASNetwork"],
    "simple_cnn": ["SimpleCnn"],
    "spnet_backbone": ["SpResNetDet"],
    "super_network": ["DartsNetwork", "CARSDartsNetwork", "GDASDartsNetwork"],
    "text_cnn": ["TextCells", "TextCNN"],
    "gcn": ["GCN"],
    "vit": ["VisionTransformer"],
    "mtm_sr": ["MtMSR"],
    "unet": ["Unet"]
})


def register_networks(backend):
    """Import and register network automatically."""
    if backend == "pytorch":
        from . import pytorch
    elif backend == "tensorflow":
        from . import tensorflow
    elif backend == "mindspore":
        from . import mindspore
