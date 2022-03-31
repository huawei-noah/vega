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

"""Defined Conf for Pipeline."""
from vega.common import ClassType
from vega.common import ConfigSerializable


class HostEvaluatorConfig(ConfigSerializable):
    """Default Evaluator config for HostEvaluator."""

    _type_name = ClassType.HOST_EVALUATOR
    type = None
    evaluate_latency = True
    cuda = True
    metric = {'type': 'accuracy'}
    report_freq = 10
    is_fusion = False

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
    opset_version = 9  # for torch model convert
    precision = 'FP32'
    cuda = False
    evaluate_latency = True
    metric = {'type': 'accuracy'}
    calculate_metric = False
    report_freq = 10
    quantize = False
    is_fusion = False
    reshape_batch_size = 1
    save_intermediate_file = False
    custom = None
    repeat_times = 10
    muti_input = True


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
