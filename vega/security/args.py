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

"""Security args."""
import os
import re
import yaml


def add_args(parser):
    """Add security args."""
    _config = parser.add_argument_group(title='security setting')
    _config.add_argument("-s", "--security", dest='security', action='store_true',
                         help="enable safe mode")
    return parser


def _check_value(value, pattern):
    if isinstance(value, str) and len(re.compile(pattern).findall(value)) > 0:
        raise ValueError("{} contains invalid characters.".format(value))


def _check_dict(dict_value, pattern):
    """Check dict."""
    if not isinstance(dict_value, dict):
        return
    for item in dict_value:
        value = dict_value[item]
        if isinstance(value, dict):
            _check_dict(value, pattern)
        else:
            _check_value(value, pattern)


def check_msg(msg):
    """Check msg."""
    _check_dict(msg, pattern="[^_A-Za-z0-9\\s:/.~-]")


def check_args(args):
    """Check args."""
    args_dict = vars(args)
    _check_dict(args_dict, pattern="[^_A-Za-z0-9:/.~-]")


def check_yml(config_yaml):
    """Check yml."""
    if config_yaml is None:
        raise ValueError("config path can't be None or empty")
    if os.stat(config_yaml).st_uid != os.getuid():
        raise ValueError(f"The file {config_yaml} not belong to the current user")
    with open(config_yaml) as f:
        raw_dict = yaml.safe_load(f)
        _check_dict(raw_dict, pattern=r"[^_A-Za-z0-9\s\<\>=\[\]\(\),!\{\}:/.~-]")


def check_job_id(job_id):
    """Check Job id."""
    if not isinstance(job_id, str):
        raise TypeError('"job_id" must be str, not {}'.format(type(job_id)))
    _check_value(job_id, pattern="[^_A-Za-z0-9]")


def check_input_shape(input_shape):
    """Check input shape."""
    if not isinstance(input_shape, str):
        raise TypeError('"input_shape" must be str, not {}'.format(type(input_shape)))
    _check_value(input_shape, pattern="[^_A-Za-z0-9:,]")


def check_out_nodes(out_nodes):
    """Check out nodes."""
    if not isinstance(out_nodes, str):
        raise TypeError('"out_nodes" must be str, not {}'.format(type(out_nodes)))
    _check_value(out_nodes, pattern="[^_A-Za-z0-9:/]")


def check_backend(backend):
    """Check backend."""
    if backend not in ["tensorflow", "caffe", "onnx", "mindspore"]:
        raise ValueError("The backend only support tensorflow, caffe, onnx and mindspore.")


def check_hardware(hardware):
    """Check hardware."""
    if hardware not in ["Davinci", "Bolt", "Kirin990_npu"]:
        raise ValueError("The hardware only support Davinci and Bolt.")


def check_precision(precision):
    """Check precision."""
    if precision.upper() not in ["FP32", "FP16"]:
        raise ValueError("The precision only support FP32 and FP16.")


def check_repeat_times(repeat_times):
    """Check repeat times."""
    MAX_EVAL_EPOCHS = 10000
    if not isinstance(repeat_times, int):
        raise TypeError('"repeat_times" must be int, not {}'.format(type(repeat_times)))
    if not 0 < repeat_times <= MAX_EVAL_EPOCHS:
        raise ValueError("repeat_times {} is not in valid range (1-{})".format(repeat_times, MAX_EVAL_EPOCHS))


def path_verify(path):
    """Verify path."""
    return re.sub(r"[^_A-Za-z0-9\/.]", "", path)
