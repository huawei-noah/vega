# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Calculate flops and parameters of a quantization model."""
import torch
import torch.nn as nn
from thop import profile
from thop.vision.basic_hooks import count_convNd, zero_ops
from .quant_conv import QuantConv


def count_quant_conv(module, input, output):
    """Calculate parameters of a quantization model.

    :param module: quantized module
    :type model: nn.Module
    :param input: input tensor list
    :type input: (torch.Tensor,)
    :param output: output tensor
    :type output: torch.Tensor
    """
    total_params = 0
    for param in module.parameters():
        total_params += param.numel()
    module.total_params[0] = total_params
    if hasattr(module, 'nbit_w'):
        module.total_params[0] = total_params * module.nbit_w
    count_convNd(module, input, output)
    if hasattr(module, 'nbit_w') and hasattr(module, 'nbit_a'):
        module.total_ops = module.total_ops * (module.nbit_w + module.nbit_a)


def cal_model_params(model, gpu_id=0, input_size=[1, 3, 224, 224]):
    """Calculate parameters of a quantization model.

    :param model: quantized pytorch model
    :type model: nn.Module
    :param gpu_id: gpu id
    :type gpu_id: int
    :param input_size: input size
    :type input_size: list
    :return: total params, params of each layer
    :rtype: int, list
    """
    input = torch.randn(input_size)
    flops, params = profile(model.cpu(), inputs=(input, ),
                            custom_ops={QuantConv: count_quant_conv,
                            nn.BatchNorm2d: zero_ops}, verbose=False)
    return params, []


def cal_model_flops(model, gpu_id=0, input_size=[1, 3, 224, 224]):
    """Calculate flops of a quantization model.

    :param model: quantized pytorch model
    :type model: nn.Module
    :param gpu_id: gpu id
    :type gpu_id: int
    :param input_size: input size
    :type input_size: list
    :return: total flops, flops of each layer
    :rtype: int, list
    """
    input = torch.randn(input_size)
    flops, params = profile(model.cpu(), inputs=(input, ),
                            custom_ops={QuantConv: count_quant_conv,
                            nn.BatchNorm2d: zero_ops}, verbose=False)
    return flops, []
