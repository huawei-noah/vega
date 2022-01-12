# -*- coding: utf-8 -*-

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

"""Quantized Convlution."""
import logging
from vega.modules.operators import ops, quant
from vega.common import ClassFactory, ClassType


@ClassFactory.register(ClassType.NETWORK)
class Quantizer(object):
    """Model Quantization class."""

    def __init__(self, model, nbit_w_list=8, nbit_a_list=8, skip_1st_layer=True):
        super().__init__()
        self.idx = 0
        self.nbit_w_list = nbit_w_list
        self.nbit_a_list = nbit_a_list
        self.skip_1st_layer = skip_1st_layer
        self.model = model

    def _next_nbit(self):
        """Get next nbit."""
        if isinstance(self.nbit_w_list, list) and isinstance(self.nbit_a_list, list):
            nbit_w, nbit_a = self.nbit_w_list[self.idx], self.nbit_a_list[self.idx]
            self.idx += 1
        else:
            nbit_w, nbit_a = self.nbit_w_list, self.nbit_a_list
        return nbit_w, nbit_a

    def _quant_conv(self, model):
        """Quantize the convolutional layer."""
        if not isinstance(model, ops.Conv2d):
            return model
        nbit_w, nbit_a = self._next_nbit()
        quant_model = quant.QuantConv(model.in_channels, model.out_channels, model.kernel_size,
                                      model.stride, model.padding, model.dilation, model.groups, model.bias)
        quant_model.build(nbit_w=nbit_w, nbit_a=nbit_a)
        return quant_model

    def __call__(self):
        """Quantize the entire model."""
        if self.nbit_w_list is None or self.nbit_a_list is None:
            logging.warning("nbit_w or nbit_a is None, model can not be quantified.")
            return self.model
        is_first_conv = True
        for name, layer in list(self.model.named_modules()):
            if not isinstance(layer, ops.Conv2d) and self.skip_1st_layer:
                continue
            if is_first_conv:
                is_first_conv = False
                continue
            if layer.groups == 1:
                quant_conv = self._quant_conv(layer)
            self.model.set_module(name, quant_conv)
        return self.model

    def custom_hooks(self):
        """Calculate flops and params."""
        return quant.quant_custom_ops()
