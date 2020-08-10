# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""model quantization wrapper."""
import tensorflow as tf
from .quant_conv import QuantConv


class Quantizer(object):
    """Model Quantization class."""

    def quant_model(self, model, nbit_w_list=8, nbit_a_list=8, skip_1st_layer=True):
        """Quantize the entire model.

        :param model: pytorch model
        :type model: nn.Module
        :param nbit_w_list: bits of weights
        :type nbit_w_list: list
        :param nbit_a_list: bits of activations
        :type nbit_a_list: list
        :param skip_1st_layer: whether skip 1st layer
        :type skip_1st_layer: bool
        :return: quantized pytorch model
        :rtype: nn.Module
        """
        if nbit_w_list is None or nbit_a_list is None:
            return model
        dummy_input = tf.random.normal((1, 32, 32, 3))
        quant_info = {}
        quant_info['skip_1st_layer'] = skip_1st_layer
        quant_info['nbit_w_list'] = nbit_w_list
        quant_info['nbit_a_list'] = nbit_a_list
        quant_info['extra_params'] = 0
        quant_info['extra_flops'] = 0
        model(dummy_input, True, quant_info)
        return model
