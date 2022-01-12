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

import tensorflow.compat.v1 as tf
from vega.modules.module import Module
from ..functions.serializable import OperatorSerializable


@tf.custom_gradient
def sign(input):
    """Apply Sign class in dorefa-net for weights.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    input = tf.math.sign(input)

    def grads(grad_output):
        return grad_output

    return input, grads


@tf.custom_gradient
def xnor(input):
    """Apply Sign class in dorefa-net for weights.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    input = tf.math.sign(input) * tf.reduce_mean(tf.math.abs(input), axis=[1, 2, 3], keepdims=True)

    def grads(grad_output):
        return grad_output

    return input, grads


@tf.custom_gradient
def scale_sign(input):
    """Apply Sign class in dorefa-net for weights.

    :param input: batch of input
    :type input: Tensor
    :return: quantized output
    :rtype: Tensor
    """
    input = tf.math.sign(input) * tf.reduce_mean(tf.math.abs(input))

    def grads(grad_output):
        return grad_output

    return input, grads


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
        w = tf.math.tanh(w)
        w = w / (2 * tf.reduce_max(tf.math.abs(w))) + 0.5
        w = 2 * quantize_w(w, nbit_w) - 1
    return w


@tf.custom_gradient
def quantize_w(input, nbit):
    """Quantization function for weights.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :return: quantized output and grad function
    :rtype: Tensor, fn
    """
    scale = tf.cast((2 ** nbit - 1), input.dtype)
    output = tf.math.round(input * scale) / scale

    def grads(grad_output):
        if grad_output is None:
            return tf.zeros_like(input), None
        return grad_output, None

    return output, grads


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
        w = quantize_w(tf.clip_by_value(w, -1, 1), nbit_w - 1)
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
    return sign(w) * tf.reduce_mean(tf.math.abs(tf.Variable(w)))


def dorefa_a(input, nbit_a, *args, **kwargs):
    """Dorefa quantization for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :return: quantized output
    :rtype: Tensor
    """
    return quantize_a(tf.clip_by_value(input, 0, 1.0), nbit_a)


@tf.custom_gradient
def quantize_a(input, nbit):
    """Quantization function for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :return: quantized output and grad function
    :rtype: Tensor, fn
    """
    scale = tf.cast((2 ** nbit - 1), input.dtype)
    output = tf.math.round(input * scale) / scale

    def grads(grad_output):
        if grad_output is None:
            return tf.zeros_like(input), None
        return grad_output, None

    return output, grads


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
    x = 0.5 * (tf.math.abs(input) - tf.math.abs(input - alpha) + alpha)
    scale = tf.cast((2 ** nbit_a - 1) if alpha is None else (2 ** nbit_a - 1) / alpha, input.dtype)
    return quantize_a(x, nbit_a, scale)


@tf.custom_gradient
def bireal_a_calc(input):
    """Forward and backward for bireal_a.

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
    input = tf.math.sign(tf.clip_by_value(input, -1, 1))

    def grads(grad_output):
        grad_input = (2 + 2 * input) * tf.cast(tf.math.less(input, 0), dtype=input.dtype) + \
                     (2 - 2 * input) * tf.cast(tf.math.greater_equal(input, 0), dtype=input.dtype)
        grad_input = tf.minimum(grad_input, 0)
        grad_input *= grad_output
        return grad_input

    return input, grads


def bireal_a(input, nbit_a=1, *args, **kwargs):
    """Adaptor for bireal_a.

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
    return bireal_a_calc(input)


class QuantConv(Module, OperatorSerializable):
    """General QuantConv class for quantized convolution.

    The params are the same as nn.Conv2d
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same', dilation=1, groups=1,
                 bias=True):
        super(QuantConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding if not isinstance(padding, int) else 'same'
        self.data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
        self.dilation = dilation
        self.group = groups
        self.bias = bias
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

    @property
    def out_channels(self):
        """Output Channel for Module."""
        return self._out_channels

    @out_channels.setter
    def out_channels(self, value):
        """Output Channel for Module."""
        self._out_channels = value

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
            self.alpha_a = tf.get_variable(self.name + '/alpha_a', initializer=10. * tf.ones((1.), dtype=tf.float32))
        else:
            self.alpha_a = None
        if quan_name_w == 'pact':
            self.alpha_w = tf.get_variable(self.name + '/alpha_w', initializer=10. * tf.ones((1.), dtype=tf.float32))
        else:
            self.alpha_w = None
        if has_offset:
            self.offset = tf.get_variable(self.name + '/offset', initializer=tf.zeross((1.), dtype=tf.float32))
        else:
            self.offset = None

    def call(self, input, *args, **kwarg):
        """Forward function of quantized convolution.

        :param input: batch of input
        :type input: Tensor
        :return: output
        :rtype: Tensor
        """
        channel_axis = 1 if (self.data_format == 'NCHW' or self.data_format == 'channels_first') else 3
        self.in_channels = int(input.get_shape()[channel_axis])
        input_size = list(input.get_shape())[1:]
        input_size.pop(channel_axis - 1)

        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.stride == 2 or self.stride == (2, 2):
                if channel_axis == 1:
                    x = tf.pad(
                        input[:, :, ::2, ::2],
                        tf.constant([[0, 0], [0, 0], [diff_channels // 2, diff_channels - diff_channels // 2]]),
                        "CONSTANT",
                        0
                    )
                else:
                    x = tf.pad(
                        input[:, ::2, ::2, :],
                        tf.constant([[0, 0], [0, 0], [diff_channels // 2, diff_channels - diff_channels // 2]]),
                        "CONSTANT",
                        0
                    )
                return x
            else:
                x = tf.pad(
                    input,
                    tf.constant([[0, 0], [0, 0], [diff_channels // 2, diff_channels - diff_channels // 2]]),
                    "CONSTANT",
                    0
                )
                return x
        # w quan
        self.weight = tf.get_variable(self.name + '/kernel',
                                      shape=[self.kernel_size[0], self.kernel_size[1],
                                             self.in_channels, self.out_channels],
                                      initializer=tf.initializers.variance_scaling(scale=1.0 / 3,
                                                                                   distribution='uniform'))
        if self.nbit_w < 32:
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.alpha_a)
        else:
            x = tf.nn.relu(input)

        x = tf.nn.conv2d(x, w, strides=self.stride, padding=self.padding.upper(), dilations=self.dilation,
                         name=self.name, data_format=self.data_format)
        return x


def quant_custom_ops():
    """Return quant custom ops."""
    return None
