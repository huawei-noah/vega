# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
    "sp_nas": ["SpNasS", "SpNasP"],
    "sr_ea": ["SRCodec", "SRMutate", "SRRandom"],
    "mfasc": ["search_algorithm:MFASC"],
    "opt_nas": ["OperatorSearchSpace", "OperatorReplaceCallback"]
})
