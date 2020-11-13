# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Model counter of FLOPS and parameters."""
from copy import deepcopy
import zeus


def add_new_hooks(custom_hooks):
    """Add new register hooks to custom hooks."""
    import torch.nn as nn
    from thop.profile import register_hooks
    from thop.vision.basic_hooks import count_softmax
    from zeus.modules.operators import ops
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

    for k, v in add_register_hooks.items():
        if k not in register_hooks and k not in custom_hooks:
            custom_hooks[k] = v
    return custom_hooks


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
    if zeus.is_torch_backend():
        from thop import profile
        try:
            _model = deepcopy(model)
        except Exception as e:
            _model = model
        if custom_hooks is None:
            custom_hooks = {}
        custom_hooks = add_new_hooks(custom_hooks)
        inputs = (input,)
        flops, params = profile(_model, inputs, custom_hooks, verbose)
        del _model
    elif zeus.is_tf_backend():
        import tensorflow.compat.v1 as tf
        with tf.Graph().as_default() as graph:
            dummy_input = tf.placeholder(dtype=tf.float32, shape=input.shape.as_list())
            model.training = False
            model(dummy_input)
            opts = tf.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.profiler.profile(graph, cmd='op', options=opts).total_float_ops
            opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
            params = tf.profiler.profile(graph, cmd='op', options=opts).total_parameters
            flops *= 0.5
    elif zeus.is_ms_backend():
        # TODO
        flops, params = 0, 0
    return flops, params
