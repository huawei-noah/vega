# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Quantized Convlution."""
import math
import tensorflow as tf


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
            return tf.zeros_like(input)
        return grad_output
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


def dorefa_a(input, nbit_a, alpha=None, offset=None):
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
    scale = tf.cast((2 ** nbit_a - 1) if alpha is None else (2 ** nbit_a - 1) / alpha, input.dtype)
    return quantize_a(tf.clip_by_value(input, 0, 1.0), nbit_a, scale)


@tf.custom_gradient
def quantize_a(input, nbit, scale):
    """Quantization function for activations.

    :param input: batch of input
    :type input: Tensor
    :param nbit: bit width
    :type nbit: int
    :param scale: calculated scale
    :type scale: float or Tensor
    :return: quantized output and grad function
    :rtype: Tensor, fn
    """
    output = tf.math.round(input * scale) / scale

    def grads(grad_output):
        if grad_output is None:
            return tf.zeros_like(input)
        return grad_output
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


class QuantConv(tf.layers.Conv2D):
    """General QuantConv class for quantized convolution.

    The params are the same as nn.Conv2d
    """

    def __init__(self, out_channels, kernel_size, name, strides=1, padding='same', dilation=1,
                 groups=1, use_bias=True, data_format='channels_first'):
        super(QuantConv, self).__init__(out_channels, kernel_size, strides, padding,
                                        data_format, dilation, use_bias=use_bias)
        self.out_channels = out_channels
        self.data_format = 'NCHW' if self.data_format == 'channels_first' else 'NHWC'
        self.group = groups
        if self.use_bias:
            self.bias = tf.get_variable(name + '/bias', initializer=tf.zeros((out_channels)))
        else:
            self.bias = None

    def quant_config(self, quan_name_w='dorefa', quan_name_a='dorefa', has_offset=False, quant_info=None, name=''):
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
        if quant_info is None:
            self.nbit_w = 1
            self.nbit_a = 1
        else:
            if isinstance(quant_info['nbit_w_list'], list) and isinstance(quant_info['nbit_a_list'], list):
                self.nbit_w, self.nbit_a = quant_info['nbit_w_list'].pop(0), quant_info['nbit_a_list'].pop(0)
            else:
                self.nbit_w, self.nbit_a = quant_info['nbit_w_list'], quant_info['nbit_a_list']
            self.quant_info = quant_info

        name_w_dict = {'dorefa': dorefa_w, 'pact': dorefa_w, 'wrpn': wrpn_w, 'xnor': xnor_w, 'bireal': bireal_w}
        name_a_dict = {'dorefa': dorefa_a, 'pact': pact_a, 'wrpn': dorefa_a, 'xnor': dorefa_a, 'bireal': bireal_a}
        self.quan_w = name_w_dict[quan_name_w]
        self.quan_a = name_a_dict[quan_name_a]
        if quan_name_a == 'pact':
            self.alpha_a = tf.get_variable(name + '/alpha_a', initializer=10. * tf.ones((1.), dtype=tf.float32))
        else:
            self.alpha_a = None
        if quan_name_w == 'pact':
            self.alpha_w = tf.get_variable(name + '/alpha_w', initializer=10. * tf.ones((1.), dtype=tf.float32))
        else:
            self.alpha_w = None
        if has_offset:
            self.offset = tf.get_variable(name + '/offset', initializer=tf.zeross((1.), dtype=tf.float32))
        else:
            self.offset = None

    def calc_flops_params(self, in_channels, out_channels, kernel_size, xW, xH):
        """Calculate extra flops and params, append in quant_info.

        :param in_channels: in_channels of input
        :type in_channels: int
        :param out_channels: out_channels of input
        :type out_channels: int
        :param kernel_size: kernel_size of input
        :type in_channels: int
        :param xW: width of input
        :type xW: int
        :param xH: height of input
        :type xH: int
        """
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            ks1, ks2 = kernel_size[0], kernel_size[1]
        else:
            ks1, ks2 = kernel_size, kernel_size
        multiplier = self.nbit_w - 1
        self.quant_info['extra_params'] += multiplier * (ks1 * ks2 * in_channels * 16)
        multiplier += self.nbit_a
        self.quant_info['extra_flops'] += multiplier * (16 * xW * xH * (in_channels * ks1 * ks2))

    def __call__(self, input):
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
        if self.quant_info:
            self.calc_flops_params(self.in_channels, self.out_channels, self.kernel_size,
                                   int(input_size[0]), int(input_size[1]))
        # 0-bit: identity mapping
        if self.nbit_w == 0 or self.nbit_a == 0:
            diff_channels = self.out_channels - self.in_channels
            if self.strides == 2 or self.strides == (2, 2):
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
                                      initializer=tf.random.normal(self.kernel_size + (self.in_channels,
                                                                                       self.out_channels,)))
        if self.nbit_w < 32:
            self.nbit_w = 1
            w = self.quan_w(self.weight, self.nbit_w, self.alpha_w, self.offset)
        else:
            w = self.weight
        # a quan
        if self.nbit_a < 32:
            x = self.quan_a(input, self.nbit_a, self.alpha_a)
        else:
            x = tf.nn.relu(input)

        if self.group == 1:
            x = tf.nn.conv2d(x, w, strides=self.strides, padding=self.padding.upper(),
                             dilations=self.dilation_rate, name=self.name,
                             data_format=self.data_format)
        else:
            x = tf.nn.depthwise_conv2d(x, w, strides=self.strides, padding=self.padding.upper(),
                                       dilations=self.dilation_rate, name=self.name,
                                       data_format=self.data_format)
        return x
