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

"""Model counter of FLOPS and parameters."""
from copy import deepcopy
import vega
import numpy as np

extension_hooks = {}


def register_extension_hooks(hooks):
    """Register extension hooks."""
    extension_hooks.update(hooks)


def add_new_hooks(custom_hooks):
    """Add new register hooks to custom hooks."""
    import torch.nn as nn
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_softmax
    from vega.modules.operators import ops
    add_register_hooks = {
        nn.PReLU: register_hooks[nn.ReLU],
        nn.ELU: register_hooks[nn.ReLU],
        nn.Softmax: count_softmax,
        ops.Conv2d: register_hooks[nn.Conv2d],
        ops.BatchNorm2d: register_hooks[nn.BatchNorm2d],
        ops.Relu: register_hooks[nn.ReLU],
        ops.Relu6: register_hooks[nn.ReLU6],
        ops.MaxPool2d: register_hooks[nn.MaxPool2d],
        ops.AdaptiveAvgPool2d: register_hooks[nn.AdaptiveAvgPool2d],
        ops.AvgPool2d: register_hooks[nn.AvgPool2d],
        ops.Linear: register_hooks[nn.Linear],
    }
    if extension_hooks:
        add_register_hooks.update(extension_hooks)
    for k, v in add_register_hooks.items():
        if k not in register_hooks and k not in custom_hooks:
            custom_hooks[k] = v
    return custom_hooks


def _do_calc_flops_params(model, input, custom_hooks=None, verbose=False):
    from thop import profile
    if custom_hooks is None:
        custom_hooks = {}
    custom_hooks = add_new_hooks(custom_hooks)
    inputs = (input,)
    flops, params = profile(model, inputs, custom_hooks, verbose)
    return flops, params


def calc_model_flops_params(model, input, custom_hooks=None, verbose=False):
    """Pytorch model flops and parameters calculation.

    :param model: pytorch model
    :type model: torch.nn.Module
    :param input: pytorch input tensor
    :type input: torch.Tensor
    :param custom_hooks: hooks defined by outside customer
    :type custom_hooks: dict or None
    :param verbose: whether to print op type which not in collection
    :type verbose: bool, default True
    :return: flops and params
    :rtype: float, float
    """
    try:
        _model = deepcopy(model)
    except Exception:
        _model = model

    if vega.is_torch_backend():
        from vega.modules.arch.architecture import register_clear_module_arch_params_hooks
        flops, params = _do_calc_flops_params(_model, input, custom_hooks, verbose)
        register_clear_module_arch_params_hooks(model)
    elif vega.is_tf_backend():
        import tensorflow.compat.v1 as tf
        with tf.Graph().as_default() as graph:
            dummy_input = tf.placeholder(dtype=tf.float32, shape=input.shape.as_list())
            _model.training = False
            _model(dummy_input)
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(graph, cmd='op', options=opts).total_float_ops
            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params = tf.profiler.profile(graph, cmd='op', options=opts).total_parameters
            flops *= 0.5
    elif vega.is_ms_backend():
        total_params = 0
        for param in model.trainable_params():
            total_params += np.prod(param.shape)
        params = total_params
        # TODO
        flops = 0

    del _model
    return flops, params
