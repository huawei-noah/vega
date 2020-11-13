# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
from zeus.common import ClassType
from zeus.common import ConfigSerializable


class GPUEvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for GPUEvaluator."""

    _type_name = ClassType.GPU_EVALUATOR
    type = None
    evaluate_latency = None
    cuda = True
    metric = {'type': 'accuracy'}
    report_freq = 10

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        check_rules = {}
        return check_rules


class DavinciMobileEvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for DavinciMobileEvaluator."""

    _type_name = ClassType.DAVINCI_MOBILE_EVALUATOR
    backend = "pytorch"
    hardware = "Davinci"
    remote_host = ""
    cuda = False
    evaluate_latency = True
    metric = {'type': 'accuracy'}
    report_freq = 10


class EvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for Evaluator."""

    _type_name = ClassType.EVALUATOR
    type = 'Evaluator'
    gpu_evaluator = GPUEvaluatorConfig
    gpu_evaluator_enable = False
    davinci_mobile_evaluator = DavinciMobileEvaluatorConfig
    davinci_mobile_evaluator_enable = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules = {"type": {"required": True, "type": str},
                 "gpu_evaluator": {"type": dict}}
        return rules
