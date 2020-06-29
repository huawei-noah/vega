# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""model quantization wrapper."""
import torch.nn as nn
from .quant_conv import QuantConv


class Quantizer:
    """Model Quantization class."""

    def __init__(self):
        self.idx = 0

    def _quant_conv(self, name, m, nbit_w_list=8, nbit_a_list=8):
        """Quantize the convolutional layer.

        :param name: name of module
        :type name: string
        :param m: pytorch model
        :type m: nn.Module
        :param nbit_w_list: bits of weights
        :type nbit_w_list: list
        :param nbit_a_list: bits of activations
        :type nbit_a_list: list
        :return: quantized pytorch model
        :rtype: nn.Module
        """
        if isinstance(m, nn.Conv2d):
            if isinstance(nbit_w_list, list) and isinstance(nbit_a_list, list):
                nbit_w, nbit_a = nbit_w_list[self.idx], nbit_a_list[self.idx]
                self.idx += 1
            else:
                nbit_w, nbit_a = nbit_w_list, nbit_a_list
            m_new = QuantConv(m.in_channels, m.out_channels, m.kernel_size,
                              m.stride, m.padding, m.dilation, m.groups, m.bias is not None)
            m_new.quant_config(nbit_w=nbit_w, nbit_a=nbit_a)
            m_new.weight.data = m.weight.data
            if m_new.bias is not None:
                m_new.bias.data = m.bias.data
            return m_new
        else:
            for name_i, m_i in m.named_children():
                m._modules[name_i] = self._quant_conv(name_i, m_i, nbit_w_list, nbit_a_list)
            return m

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
        self.idx = 0
        for name, m in model.named_children():
            if skip_1st_layer and name == 'conv1':
                continue
            model._modules[name] = self._quant_conv(name, m, nbit_w_list, nbit_a_list)
        return model
