# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Custom functions of tensorflow."""
import math
import tensorflow.compat.v1 as tf
from zeus.common.config import Config
from zeus.common.class_factory import ClassType, ClassFactory
from zeus.modules.operators.functions.serializable import OperatorSerializable
from zeus.common.general import General

enable_scope_name = True


class Module(object):
    """Base Module to adapter tf Module."""

    data_format = 'channels_first'

    def __init__(self):
        self.parent_scope_name = ''
        self._scope_name = ''
        self._modules = Config()
        self._training = True
        self.enable_scope_name = enable_scope_name
        self.data_format = General.data_format

    def add_module(self, name, model):
        """Add models into self._models."""
        setattr(self, str(name), model)

    def named_modules(self):
        """Return names spaces."""
        _names_modules = []
        for model in self.children():
            if isinstance(model, Module):
                _names_modules.append(((model._scope_name, model)))
                child_modules = model.named_modules()
                _names_modules.extend(child_modules)
        return _names_modules

    def named_children(self):
        """Return names children."""
        return [(name, module) for name, module in self._modules.items()]

    def children(self):
        """Get child models of current Module."""
        for model in self._modules.values():
            if isinstance(model, Module):
                model._scope_name = "{}.{}".format(
                    self._scope_name, model.parent_scope_name) if self._scope_name else model.parent_scope_name
            yield model

    @property
    def training(self):
        """Get training flag."""
        return self._training

    @training.setter
    def training(self, value):
        """Set training flag."""
        self._training = value
        for module in self.children():
            module.training = value

    def __setattr__(self, key, value):
        """Set name to modules."""
        self.__dict__[key] = value
        if isinstance(value, Module):
            if self.enable_scope_name:
                value.parent_scope_name = key
            self._modules[key] = value

    def __getattribute__(self, name):
        """Get modules by name."""
        value = object.__getattribute__(self, name)
        if isinstance(value, Module) and self.enable_scope_name:
            value._scope_name = "{}.{}".format(
                self._scope_name, value.parent_scope_name) if self._scope_name else value.parent_scope_name
        return value

    def set_parameters(self, name, value):
        """Set Parameters."""
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            setattr(self, name, tf.get_variable(name, initializer=value))

    def get_weights(self, name):
        """Get weights by name."""
        return tf.get_default_graph().get_tensor_by_name('{}:0'.format(name))

    def get_weight_ops(self, name):
        """Get weight ops."""
        all_weight = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        weight_ops = [t for t in all_weight if not t.name.startswith(name)]
        return weight_ops

    def call(self, inputs, *args, **kwarg):
        """Call inputs."""
        output = inputs
        for model in self.children():
            output = model(output)
        return output

    def __call__(self, inputs, *args, **kwargs):
        """Call call function."""
        return self.call(inputs, *args, **kwargs)

    def modules(self):
        """Get the current modules."""
        if self._modules.values():
            return self._modules.values()
        else:
            return [self]


class He_initial(object):
    """Initialize of Hekaiming."""

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, tensor, **kwargs):
        """Call He_initial function."""
        c, h, w = get_shape(tensor)[1:]
        fan_in = c * h * w
        std = math.sqrt(2) / math.sqrt(fan_in)
        initializer = tf.random_normal_initializer(0, std * self.scale)
        return initializer


class Initial(object):
    """Initialize of Hekaiming."""

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, tensor, **kwargs):
        """Call initial function."""
        return tf.variance_scaling_initializer()


@ClassFactory.register(ClassType.NETWORK)
class QuantizeConv2d(OperatorSerializable):
    """QuantizeConv2d Module inherit nn.Module."""

    def __init__(self):
        """Construct Identity class."""
        OperatorSerializable.__init__(self)

    def __call__(self, inputs, **kwargs):
        """Call QuantizeConv2d function."""
        # todo
        return inputs


@ClassFactory.register(ClassType.NETWORK)
class Conv2d(Module, OperatorSerializable):
    """Fuse and unified conv2d args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, groups=1,
                 dilation=1, separable=False, depthwise=False, padding_mode='same'):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.dilation = dilation
        self.kernel_initial = Initial()
        self.bias_initial = tf.zeros_initializer()
        self.name = None
        self.reuse = None
        self.separable = separable
        self.depthwise = depthwise
        self.padding_mode = padding_mode

    def __call__(self, inputs, **kwargs):
        """Call separable_conv2d function."""
        initializer = self.kernel_initial(inputs)
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            if self.dilation > 1:
                return tf.keras.layers.SeparableConv2D(filters=self.out_channels,
                                                       kernel_size=self.kernel_size,
                                                       strides=self.stride,
                                                       data_format=self.data_format,
                                                       dilation_rate=self.dilation,
                                                       padding=self.padding_mode,
                                                       use_bias=self.bias,
                                                       name='Conv2d')(inputs=inputs)
            else:
                return tf.keras.layers.Conv2D(filters=self.out_channels,
                                              kernel_size=self.kernel_size,
                                              kernel_initializer=initializer,
                                              bias_initializer=self.bias_initial,
                                              strides=self.stride,
                                              data_format=self.data_format,
                                              dilation_rate=self.dilation,
                                              padding=self.padding_mode,
                                              use_bias=self.bias,
                                              name='Conv2d')(inputs=inputs)

    def initial(self, kernel_mode='he', bias_mode='zero', kernel_scale=1., bias_scale=1.):
        """Initialize weight and bias."""
        if kernel_mode == 'he':
            self.kernel_initial = He_initial(kernel_scale)
        if bias_mode == 'zero':
            self.bias_initial = tf.zeros_initializer()


@ClassFactory.register(ClassType.NETWORK)
class SeparableConv2d(Module, OperatorSerializable):
    """Separable Conv2d args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.dilation = dilation

    def __call__(self, input, **kwargs):
        """Call separable_conv2d function."""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.keras.layers.SeparableConv2D(filters=self.out_channels,
                                                   kernel_size=self.kernel_size,
                                                   strides=self.stride,
                                                   data_format=self.data_format,
                                                   dilation_rate=self.dilation,
                                                   depthwise_initializer=tf.variance_scaling_initializer(),
                                                   pointwise_initializer=tf.variance_scaling_initializer(),
                                                   padding='SAME', use_bias=self.bias,
                                                   name='SeparableConv2d',
                                                   reuse=self.reuse)(inputs=input)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(Module, OperatorSerializable):
    """Fuse and unified MaxPool2d args."""

    def __init__(self, kernel_size, stride, padding=0):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, input, **kwargs):
        """Call MaxPooling2D function."""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.layers.MaxPooling2D(pool_size=self.kernel_size, strides=self.stride,
                                          data_format=self.data_format, padding='SAME', name='MaxPool2d')(input)


@ClassFactory.register(ClassType.NETWORK)
class Zero(Module, OperatorSerializable):
    """Class of Zero operation."""

    def __init__(self, stride):
        """Init Zero."""
        super(Zero, self).__init__()
        self.stride = stride

    def __call__(self, x, **kwargs):
        """Forward Function fo Zero."""
        if self.stride == 1:
            return tf.zeros_like(x)
        if self.data_format == 'channels_first':
            return tf.zeros_like(x)[:, :, ::self.stride, ::self.stride]
        else:
            return tf.zeros_like(x)[:, ::self.stride, ::self.stride, :]


@ClassFactory.register(ClassType.NETWORK)
class View(Module, OperatorSerializable):
    """Call squeeze."""

    def __init__(self, size=None):
        super(View, self).__init__()
        self.size = size

    def __call__(self, inputs, **kwargs):
        """Call squeeze function."""
        if not self.size:
            total_shape = 1
            for _shape in inputs.get_shape()[1:]:
                total_shape *= _shape
            return tf.reshape(inputs, [-1, total_shape])
        else:
            self.size = list(self.size)
            return tf.reshape(inputs, self.size)


@ClassFactory.register(ClassType.NETWORK)
class Relu(Module, OperatorSerializable):
    """Call relu."""

    def __init__(self, inplace=False):
        super(Relu, self).__init__()
        self.inplace = inplace

    def __call__(self, input, **kwargs):
        """Call relu function."""
        return tf.nn.relu(input)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(Module, OperatorSerializable):
    """Call relu6."""

    def __init__(self, inplace=False):
        super(Relu6, self).__init__()
        self.inplace = inplace

    def __call__(self, input, **kwargs):
        """Call relu6 function."""
        return tf.nn.relu6(input)


@ClassFactory.register(ClassType.NETWORK)
class Hswish(Module, OperatorSerializable):
    """Call Hswish."""

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def __call__(self, input, **kwargs):
        """Call Hswish function."""
        return input * tf.nn.relu6(input + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Hsigmoid(Module, OperatorSerializable):
    """Call Hsigmoid."""

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def __call__(self, input, **kwargs):
        """Call Hsigmoid function."""
        return tf.nn.relu6(input + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class AdaptiveAvgPool2d(Module, OperatorSerializable):
    """Call reduce_mean."""

    def __init__(self, output_size=(1, 1)):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def __call__(self, input, **kwargs):
        """Call reduce_mean function."""
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        return tf.reduce_mean(input, axes, keepdims=True)


@ClassFactory.register(ClassType.NETWORK)
class Linear(Module, OperatorSerializable):
    """Call dense."""

    def __init__(self, in_features, out_features, use_bias=True, activation=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

    def __call__(self, input, **kwargs):
        """Call dense function."""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.keras.layers.Dense(units=self.out_features, use_bias=self.use_bias, name='Linear',
                                         activation=self.activation)(inputs=input)


@ClassFactory.register(ClassType.NETWORK)
class AvgPool2d(Module, OperatorSerializable):
    """Call average_pooling2d."""

    def __init__(self, kernel_size, stride, padding=0, count_include_pad=True):
        super(AvgPool2d, self).__init__()
        if not stride:
            stride = kernel_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.count_include_pad = count_include_pad

    def __call__(self, input, **kwargs):
        """Call average_pooling2d function."""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.keras.layers.AveragePooling2D(pool_size=self.kernel_size,
                                                    strides=self.stride,
                                                    data_format=self.data_format,
                                                    padding='SAME',
                                                    name='AvgPool2d')(input)


@ClassFactory.register(ClassType.NETWORK)
class BatchNorm2d(Module, OperatorSerializable):
    """Call batch_normalization."""

    def __init__(self, num_features, eps=1e-05, momentum=0.997, affine=None):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = affine if affine is not None else self.training
        self.affine = affine

    def __call__(self, input, **kwargs):
        """Call batch_normalization function."""
        with tf.variable_scope(self._scope_name, reuse=tf.AUTO_REUSE):
            return tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                      axis=1 if self.data_format == 'channels_first' else 3,
                                                      epsilon=self.eps,
                                                      center=True, scale=True, fused=True,
                                                      name='BatchNorm2d')(inputs=input, training=self.training)


@ClassFactory.register(ClassType.NETWORK)
class Identity(Module, OperatorSerializable):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def __call__(self, x, **kwargs):
        """Forward function of Identity."""
        return tf.identity(x)


@ClassFactory.register(ClassType.NETWORK)
class PixelShuffle(Module, OperatorSerializable):
    """Class of PixelShuffle."""

    def __init__(self, upscale):
        super(PixelShuffle, self).__init__()
        self.upscale = upscale

    def __call__(self, inputs, **kwargs):
        """Forward function of PixelShuffle."""
        inputs = tf.cast(inputs, tf.float16)
        if self.data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 2, 3, 1])
        outputs = tf.nn.depth_to_space(inputs, self.upscale, data_format='NHWC')
        if self.data_format == 'channels_first':
            outputs = tf.transpose(outputs, [0, 3, 1, 2])
        outputs = tf.cast(outputs, tf.float32)
        return outputs


@ClassFactory.register(ClassType.NETWORK)
class Split(Module, OperatorSerializable):
    """Class of Split."""

    def __init__(self, size=None, dim=0):
        super(Split, self).__init__()
        self.size = size
        self.dim = dim

    def __call__(self, inputs, **kwargs):
        """Forward function of Split."""
        length = inputs.shape[self.dim]
        number = length // self.size
        return tf.split(inputs, number, self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Squeeze(Module, OperatorSerializable):
    """Class of Squeeze."""

    def __init__(self, dim=0):
        self.dim = dim
        super(Squeeze, self).__init__()

    def __call__(self, inputs, **kwargs):
        """Forward function of squeeze."""
        return tf.squeeze(inputs, [self.dim])


@ClassFactory.register(ClassType.NETWORK)
class Permute(Module, OperatorSerializable):
    """Class of Permute."""

    def __init__(self, size=None):
        super(Permute, self).__init__()
        self.size = size

    def __call__(self, inputs, **kwargs):
        """Forward function of Permute."""
        return tf.transpose(inputs, self.size)


@ClassFactory.register(ClassType.NETWORK)
class Stack(Module, OperatorSerializable):
    """Class of Stack."""

    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def __call__(self, inputs, **kwargs):
        """Forward function of Stack."""
        return tf.stack(inputs, self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Transpose(Module, OperatorSerializable):
    """Class of Transpose."""

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1, self.dim2 = dim1, dim2

    def __call__(self, inputs, **kwargs):
        """Call Transpose."""
        new_dim = [i for i in range(len(inputs.shape))]
        new_dim[self.dim1], new_dim[self.dim2] = new_dim[self.dim2], new_dim[self.dim1]
        return tf.transpose(inputs, new_dim)


@ClassFactory.register(ClassType.NETWORK)
class LeakyReLU(Module, OperatorSerializable):
    """Class of LeakyReLU."""

    def __init__(self, inplace=False, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.inplace = inplace
        self.alpha = negative_slope

    def __call__(self, input, **kwargs):
        """Call LeakyReLU."""
        return tf.nn.leaky_relu(input, self.alpha)


@ClassFactory.register(ClassType.NETWORK)
class InterpolateScale(Module, OperatorSerializable):
    """Upsample of torch with scale_factor."""

    def __init__(self, scale_factor=None, size=None, mode='bilinear', align_corners=False):
        super(InterpolateScale, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.size = size

    def __call__(self, inputs, **kwargs):
        """Call InterpolateScale."""
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        if self.size is not None:
            if isinstance(self.size, int):
                self.size = (self.size, self.size)
            output = tf.image.resize(inputs, size=self.size, method=self.mode, align_corners=self.align_corners)
        else:
            output = tf.image.resize_images(inputs, [inputs.shape[1] * self.scale_factor,
                                                     inputs.shape[2] * self.scale_factor], method=self.mode,
                                            align_corners=self.align_corners)
        return tf.transpose(output, [0, 3, 1, 2])


@ClassFactory.register(ClassType.NETWORK)
class MeanShift(Module, OperatorSerializable):
    """Subtract or add rgb_mean to the image."""

    def __init__(self, rgb_range, rgb_mean, rgb_std=(1.0, 1.0, 1.0), sign=-1):
        """Construct the class MeanShift.

        :param rgb_range: range of tensor, usually 1.0 or 255.0
        :param rgb_mean: mean of rgb value
        :param rgb_std: std of rgb value
        :param sign: -1 for subtract, 1 for add
        """
        super(MeanShift, self).__init__()
        self.rgb_std = rgb_std
        self.rgb_mean = rgb_mean
        self.sign = sign
        self.rgb_range = rgb_range

    def __call__(self, inputs, *args, **kwargs):
        """Call MeanShift."""
        std = tf.convert_to_tensor(self.rgb_std, dtype=tf.float32)
        self.weight = tf.eye(3)
        self.weight = tf.div(self.weight, std)
        self.bias = self.sign * self.rgb_range * tf.convert_to_tensor(self.rgb_mean, dtype=tf.float32)
        self.bias = tf.div(self.bias, std)
        res = tf.einsum('ij, njhw->nihw', self.weight, inputs)
        res = tf.transpose(res, [0, 2, 3, 1])
        res = tf.nn.bias_add(res, self.bias)
        res = tf.transpose(res, [0, 3, 1, 2])
        return res


@ClassFactory.register(ClassType.NETWORK)
class GlobalMaxPool1d(Module):
    """Construct the class GlobalMaxPool1d."""

    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def call(self, inputs, *args, **kwargs):
        """Call max_pool1d function."""
        return tf.layers.MaxPooling1D(pool_size=get_shape(inputs)[2])(inputs)


@ClassFactory.register(ClassType.NETWORK)
class MoudleList(Module, OperatorSerializable):
    """Class of LeakyReLU."""

    def __init__(self):
        super(MoudleList, self).__init__()
        self.moudle_list = []

    def append(self, moudle):
        """Append new moudle."""
        index = len(self.moudle_list)
        self.add_module('moudle_list_' + str(index), moudle)
        self.moudle_list.append(moudle)
        return self

    def __getitem__(self, idx):
        """Get item by idx."""
        return list(self.children())[idx]


def concat(inputs, dim=1):
    """Call concat according to backends."""
    if dim != 1:
        return tf.concat(inputs, axis=dim)
    if General.data_format == "channels_first":
        dim = 1
    elif General.data_format == "channels_last":
        dim = 3
    return tf.concat(inputs, axis=dim)


def mul(a, b):
    """Call mul according to backends."""
    return tf.multiply(a, b)


def random_normal(*size):
    """Apply random values from a normal distribution."""
    return tf.random.normal(size)


def softmax(input, dim=None):
    """Apply a softmax function."""
    return tf.nn.softmax(input, dim)


def gumbel_softmax_sample(input, temperature, eps=1e-20):
    """Draw a sample from the Gumbel-Softmax distribution."""
    shape = tf.shape(input)
    U = tf.random_uniform(shape, minval=0, maxval=1)
    U = -tf.log(-tf.log(U + eps) + eps)
    y = input + U
    return tf.nn.softmax(y / temperature)


def gumbel_softmax(input, dim=-1, tau=1, hard=True, eps=1e-20):
    """Apply a gumbel-softmax function."""
    keep_dims = True if dim == -1 else False
    y = gumbel_softmax_sample(input, tau, eps)
    if hard:
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)), y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y


def to_numpy(input):
    """Apply numpy function."""
    return input


def mean(input):
    """Apply mean function."""
    return tf.reduce_mean(input, [-2, -1], keepdims=True)


def pad(inputs, position):
    """Apply pad function."""
    len_dim = len(get_shape(inputs))
    pos = [[0, 0] for i in range(len_dim)]
    for i in range(len(position)):
        if i % 2 == 0:
            pos[(-(i // 2) - 1)][0] = position[i]
        else:
            pos[(-(i // 2) - 1)][1] = position[i]
    return tf.pad(inputs, pos)


def tensor_abs(inputs):
    """Apply abs function."""
    return tf.abs(inputs)


def mean_all(inputs):
    """Apply mean_all function."""
    return tf.math.reduce_mean(inputs)


def interpolate(input, size, mode='bilinear', align_corners=False):
    """Apply interpolate function."""
    x = tf.image.resize(tf.transpose(input, [0, 2, 3, 1]),
                        size=size, method=mode, align_corners=align_corners)
    x = tf.transpose(x, [0, 3, 1, 2])
    return x


def add_n(input):
    """Apply sum function."""
    return tf.add_n(list(input))


def get_shape(inputs):
    """Get shape."""
    return inputs.get_shape().as_list()


def drop_path(x, prob):
    """Drop path operation.

    :param x: input feature map
    :type x: torch tensor
    :param prob: dropout probability
    :type prob: float
    :return: output feature map after dropout
    :rtype: torch tensor
    """
    if prob <= 0.:
        return x
    keep = 1. - prob

    bernoulli_random = tf.random.uniform([int(x.get_shape()[0]), 1, 1, 1])
    mask = tf.cast(bernoulli_random < keep, tf.float32)
    x = tf.div(x, keep)
    x = tf.multiply(x, mask)
    return x


def zeros(shape):
    """Create zeros like shape."""
    res = tf.zeros(shape)
    res = tf.cast(res, tf.float32)
    return res


def maximum(arg1, arg2):
    """Get max item."""
    return tf.maximum(arg1, arg2)


def minimum(arg1, arg2):
    """Get min item."""
    return tf.minimum(arg1, arg2)


def new_constant(tensor, size, value, dtype='long'):
    """Return new tensor with shape."""
    if dtype == 'long':
        dtype = tf.float32
    elif dtype == 'uint8':
        dtype = tf.int32
    else:
        dtype = None
    if not isinstance(size, list):
        size = list(size)
    return tf.constant(value=value, dtype=dtype, shape=size)


def argmax(tensor, dim):
    """Get max and ind from dim."""
    return tf.argmax(tensor, axis=dim)


def clamp(x, min=float("-inf"), max=float("inf")):
    """Cet value after clamp."""
    return tf.clip_by_value(x, min=min, max=max)


def where(cond):
    """Return index by condition."""
    return tf.where(cond)


def unique(inputs):
    """Return the unique elements of the input tensor."""
    return tf.unique(inputs)


def log(inputs):
    """Return the log of the input tensor."""
    return tf.math.log(inputs)


def convert_to_tensor(narray, device):
    """Convert numpy to tensor."""
    return tf.convert_to_tensor(narray, tf.float32)


def new_ones(tensor, size, dtype=None):
    """Return new tensor with shape."""
    if dtype == 'long':
        dtype = tf.float32
    elif dtype == 'uint8':
        dtype = tf.int32
    else:
        dtype = None
    tf.constant(value=1, dtype=dtype, shape=size)


def arange(left, right, dtype, device):
    """Rreange from left to right."""
    if dtype == 'long':
        dtype = tf.float32
    elif dtype == 'uint8':
        dtype = tf.int32
    else:
        dtype = None
    return tf.range(left, right, dtype=dtype)


def compare_where(cond, x, y):
    """Return item by condition."""
    return tf.where(cond, x, y)


def unsqueeze(inputs, dim):
    """Expand in dim."""
    return tf.expand_dims(inputs, dim)


def expand_as(inputs, tensor):
    """Expand as tensor."""
    return tf.broadcast_to(inputs, tensor.get_shape())


def exp(tensor):
    """Return exp(tensor)."""
    return tf.math.exp(tensor)


def pow(input, exponent, out=None):
    """Calculate the exponent value of the input by element and returns the result tensor."""
    return tf.pow(input)


def ones(input_size, out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return tf.ones(input_size, out)


def one_hot(inputs, num_classes):
    """Take LongTensor with index values of shape."""
    return tf.one_hot(inputs, num_classes)


def to(input, dtype):
    """Convert input to dtype."""
    if dtype == 'long':
        dtype = tf.long
    elif dtype == 'uint8':
        dtype = tf.uint8
    elif dtype == 'float32':
        dtype = tf.float32
    return tf.cast(input, dtype=dtype)


def reduce_sum(input, dim=0, dtype=None):
    """Apply sum function."""
    out = tf.reduce_sum(input, axis=dim)
    if dtype is not None:
        out = to(out, dtype)
    return out
