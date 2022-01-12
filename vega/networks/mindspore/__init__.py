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

"""Lazy import mindspore network."""

from vega.common.class_factory import ClassFactory


ClassFactory.lazy_register("vega.networks.mindspore", {
    "dnet": ["DNet"],
    "super_network": ["DartsNetwork", "CARSDartsNetwork", "GDASDartsNetwork"],
    "backbones.load_official_model": ["OffcialModelLoader"],
    "backbones.resnet_ms": ["ResNetMs"],
    "losses.mix_auxiliary_loss": ["MixAuxiliaryLoss"],
    "faster_rcnn.faster_rcnn_resnet": ["Faster_Rcnn_MD"]
})
