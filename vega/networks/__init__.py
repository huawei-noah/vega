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

"""Import and register network automatically."""

from vega.common.class_factory import ClassFactory
from .network_desc import NetworkDesc

ClassFactory.lazy_register("vega.networks", {
    "adelaide": ["AdelaideFastNAS"],
    "bert": ["BertClassification", "TinyBertForPreTraining", "BertClassificationHeader"],
    "dnet": ["DNet", "DNetBackbone", "EncodedBlock"],
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
    "unet": ["Unet"],
    "decaug": ["DecAug"],
})


def register_networks(backend):
    """Import and register network automatically."""
    if backend == "pytorch":
        from . import pytorch
    elif backend == "tensorflow":
        from . import tensorflow
    elif backend == "mindspore":
        from . import mindspore
