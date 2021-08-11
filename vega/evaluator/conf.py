# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined Conf for Pipeline."""
from vega.common import ClassType
from vega.common import ConfigSerializable


class HostEvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for HostEvaluator."""

    _type_name = ClassType.HOST_EVALUATOR
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


class DeviceEvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for DeviceEvaluator."""

    _type_name = ClassType.DEVICE_EVALUATOR
    backend = "pytorch"
    hardware = "Davinci"
    remote_host = ""
    intermediate_format = "onnx"  # for torch model convert
    cuda = False
    evaluate_latency = True
    metric = {'type': 'accuracy'}
    calculate_metric = False
    report_freq = 10
    quantize = False


class EvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for Evaluator."""

    _type_name = ClassType.EVALUATOR
    type = 'Evaluator'
    host_evaluator = HostEvaluatorConfig
    host_evaluator_enable = False
    device_evaluator = DeviceEvaluatorConfig
    device_evaluator_enable = False

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        rules = {"type": {"required": True, "type": str},
                 "host_evaluator": {"type": dict}}
        return rules
