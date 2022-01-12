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

"""Lazy import nas algorithms."""

from vega.common.class_factory import ClassFactory

ClassFactory.lazy_register("vega.algorithms.nas", {
    "adelaide_ea": ["AdelaideCodec", "AdelaideMutate", "AdelaideRandom", "AdelaideEATrainerCallback"],
    "auto_lane": ["AutoLaneNas", "AutoLaneNasCodec", "AutoLaneTrainerCallback"],
    "backbone_nas": ["BackboneNasCodec", "BackboneNasSearchSpace", "BackboneNas"],
    "cars": ["CARSAlgorithm", "CARSTrainerCallback", "CARSPolicyConfig"],
    "darts_cnn": ["DartsCodec", "DartsFullTrainerCallback", "DartsNetworkTemplateConfig", "DartsTrainerCallback"],
    "dnet_nas": ["DblockNasCodec", "DblockNas", "DnetNasCodec", "DnetNas"],
    "esr_ea": ["ESRCodec", "ESRTrainerCallback", "ESRSearch"],
    "fis": ["AutoGateGrdaS1TrainerCallback", "AutoGateGrdaS2TrainerCallback", "AutoGateS1TrainerCallback",
            "AutoGateS2TrainerCallback", "AutoGroupTrainerCallback", "CtrTrainerCallback"],
    "mfkd": ["MFKD1", "SimpleCnnMFKD"],
    "modnas": ["ModNasAlgorithm", "ModNasTrainerCallback"],
    "segmentation_ea": ["SegmentationCodec", "SegmentationEATrainerCallback", "SegmentationNas"],
    "sgas": ["SGASTrainerCallback"],
    "sm_nas": ["SmNasCodec", "SMNasM"],
    "sp_nas": ["SpNasS", "SpNasP", "ReignitionCallback"],
    "sr_ea": ["SRCodec", "SRMutate", "SRRandom"],
    "mfasc": ["search_algorithm:MFASC"],
    "opt_nas": ["OperatorSearchSpace", "OperatorReplaceCallback"],
    "dag_block_nas": ["DAGBlockNas"],
})
