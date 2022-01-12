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
import math
import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
from thop.vision.basic_hooks import count_convNd, zero_ops


class Quantizer(Function):
    """Quantize class for weights and activations.

    take a real value x in alpha*[0,1] or alpha*[-1,1]
    output a discrete-valued x in alpha*{0, 1/(2^k-1), ..., (2^k-1)/(2^k-1)} or likeness
    where k is nbit
    """

    @staticmethod
    def forward(ctx, input, nbit, alpha=None, offset=None):
        """Forward.

        :param input: batch of input
        :type input: Tensor
        :param nbit: bit width
        :type nbit: int
        :param alpha: scale factor
        :type alpha: float or Tensor
        :param offset: offset factor
        :type offset: float or Tensor
        :return: quantized output
        :rtype: Tensor
        """
        ctx.alpha = alpha
        ctx.offset = offset
        scale = (2 ** nbit - 1) if alpha is None else (2 ** nbit - 1) / alpha
        ctx.scale = scale
        return torch.round(input * scale) / scale if offset is None \
            else (torch.round(input * scale) + torch.round(offset)) / scale

    @staticmethod
    def backward(ctx, grad_output):
        """Backward.

        :param grad_output: grad of output
        :type grad_output: Tensor
        :return: grad of inputs
        :rtype: Tensor, None, None, Tensor
        """
        if ctx.offset is None:
            return grad_output, None, None, None
        else:
            return grad_output, None, None, torch.sum(grad_output) / ctx.scale


def quantize(input, nbit, alpha=None, offset=None):
    """Apply Quantize class for weights and activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return Quantizer.apply(input, nbit, alpha, offset)


class Signer(Function):
    """Sign class with STE, take the real value x, output sign(x)."""

    @staticmethod
    def forward(ctx, input):
        """Forward.

        :param input: batch of input
        :type input: Tensor
        :return: quantized output
        :rtype: Tensor
        """
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward.

        :param grad_output: grad of output
        :type grad_output: Tensor
        :return: grad of input
        :rtype: Tensor
        """
        return grad_output


def sign(input):
    """Apply Sign class with STE.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return Signer.apply(input)


class Xnor(Function):
    """Sign class in xnor-net for weights, take the real value x, output sign(x_c) * E(|x_c|)."""

    @staticmethod
    def forward(ctx, input):
        """Forward.

        :param input: batch of input
        :type input: Tensor
        :return: quantized output
        :rtype: Tensor
        """
        return torch.sign(input) * torch.mean(torch.abs(input), dim=[1, 2, 3], keepdim=True)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward.

        :param grad_output: grad of output
        :type grad_output: Tensor
        :return: grad of input
        :rtype: Tensor
        """
        return grad_output


def xnor(input):
    """Apply Sign class in xnor-net for weights.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return Xnor.apply(input)


class ScaleSigner(Function):
    """Sign class in dorefa-net for weights, take the real value x, output sign(x) * E(|x|)."""

    @staticmethod
    def forward(ctx, input):
        """Forward.

        :param input: batch of input
        :type input: Tensor
        :return: quantized output
        :rtype: Tensor
        """
        return torch.sign(input) * torch.mean(torch.abs(input))

    @staticmethod
    def backward(ctx, grad_output):
        """Backward.

        :param grad_output: grad of output
        :type grad_output: Tensor
        :return: grad of input
        :rtype: Tensor
        """
        return grad_output


def scale_sign(input):
    """Apply Sign class in dorefa-net for weights.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return ScaleSigner.apply(input)


def dorefa_w(w, nbit_w, *args, **kwargs):
    """Dorefa quantization for weights.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = torch.tanh(w)
        w = w / (2 * torch.max(torch.abs(w))) + 0.5
        w = 2 * quantize(w, nbit_w) - 1
    return w


def wrpn_w(w, nbit_w, *args, **kwargs):
    """Wrpn quantization for weights.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    if nbit_w == 1:
        w = scale_sign(w)
    else:
        w = quantize(torch.clamp(w, -1, 1), nbit_w - 1)
    return w


def xnor_w(w, nbit_w=1, *args, **kwargs):
    """Xnor quantization for weights.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in XNOR-Net.')
    return xnor(w)


def bireal_w(w, nbit_w=1, *args, **kwargs):
    """Bireal quantization for weights.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    if nbit_w != 1:
        raise ValueError('nbit_w must be 1 in Bi-Real-Net.')
    return sign(w) * torch.mean(torch.abs(w.clone().detach()))


def dorefa_a(input, nbit_a, *args, **kwargs):
    """Dorefa quantization for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return quantize(torch.clamp(input, 0, 1.0), nbit_a, *args, **kwargs)


def pact_a(input, nbit_a, alpha, *args, **kwargs):
    """PACT quantization for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    x = 0.5 * (torch.abs(input) - torch.abs(input - alpha) + alpha)
    return quantize(x, nbit_a, alpha, *args, **kwargs)


class BirealActivation(Function):
    """Bi-real sign class for activations, take the real value x, output sign(x)."""

    @staticmethod
    def forward(ctx, input, nbit_a=1):
        """Forward.

        :param input: batch of input
        :type input: Tensor
        :param nbit_a: bit width
        :type nbit_a: int
        :return: quantized output
        :rtype: Tensor
        """
        ctx.save_for_backward(input)
        return input.clamp(-1, 1).sign()

    @staticmethod
    def backward(ctx, grad_output):
        """Backward.

        :param grad_output: grad of output
        :type grad_output: Tensor
        :return: grad of input
        :rtype: Tensor
        """
        input, = ctx.saved_tensors
        grad_input = (2 + 2 * input) * input.lt(0).float() + (2 - 2 * input) * input.ge(0).float()
        grad_input = torch.clamp(grad_input, 0)
        grad_input *= grad_output
        return grad_input, None


def bireal_a(input, nbit_a=1, *args, **kwargs):
    """Apply bi-real sign class for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param alpha: scale factor
    :type alpha: float or Tensor
    :param offset: offset factor
    :type offset: float or Tensor
    :return: quantized output
    :rtype: Tensor
    """
    return BirealActivation.apply(input)


class QuantConv(nn.Conv2d):
    """General QuantConv class for quantized convolution.

    The params are the same as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(QuantConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

        if bias is True:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_custome_parameters()
        self.build()

    def build(self, nbit_a=8, nbit_w=8, quan_name_w='dorefa', quan_name_a='dorefa', has_offset=False):
        """Config the quantization settings.

        :param quan_name_w: type of weight quantization
        :type quan_name_w: string
        :param quan_name_a: type of activation quantization
        :type quan_name_a: string
        :param nbit_w: bit width of weight quantization
        :type nbit_w: int
        :param nbit_a: bit width of activation quantization
        :type nbit_a: int
        :param has_offset: whether use offset
        :type has_offset: bool
        """
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w, 'wrpn': wrpn_w, 'xnor': xnor_w, 'bireal': bireal_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a, 'wrpn': dorefa_a, 'xnor': dorefa_a, 'bireal': bireal_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]

        if quan_name_a == 'pact':
            self.alpha_a = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_a', None)
        if quan_name_w == 'pact':
            self.alpha_w = nn.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_parameter('alpha_w', None)
        if has_offset:
            self.offset = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('offset', None)

        if self.alpha_a is not None:
            nn.init.constant_(self.alpha_a, 10)
        if self.alpha_w is not None:
            nn.init.constant_(self.alpha_w, 10)
        if self.offset is not None:
            nn.init.constant_(self.offset, 0)

    def reset_custome_parameters(self):
        """Reset the parameters customely."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def forward(self, input):
        """Forward function of quantized convolution.

        :param input: batch of input
        :type input: Tensor
        :return: output
        :rtype: Tensor
        """
        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                x = F.pad(
                    input[:, :, ::2, ::2],
                    (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2),
                    'constant',
                    0
                )
                return x
            else:
                x = F.pad(
                    input,
                    (0, 0, 0, 0, diff_channels // 2, diff_channels - diff_channels // 2),
                    'constant',
                    0
                )
                return x
        # w quan
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.alpha_a)
        else:
            x = F.relu(input)
        x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


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
        module.total_params[0] = total_params * module.nbit_w / 32
    count_convNd(module, input, output)
    if hasattr(module, 'nbit_w') and hasattr(module, 'nbit_a'):
        module.total_ops = module.total_ops * (module.nbit_w / 32 + module.nbit_a / 32)


def quant_custom_ops():
    """Return quant custom ops."""
    custom_ops = {QuantConv: count_quant_conv,
                  nn.BatchNorm2d: zero_ops}
    return custom_ops
