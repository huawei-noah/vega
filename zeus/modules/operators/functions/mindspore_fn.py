# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Custom functions of pytorch."""
import math
from functools import reduce
import mindspore.nn as nn
import mindspore
import numpy as np
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer
from .serializable import OperatorSerializable
from zeus.common.class_factory import ClassType, ClassFactory
from zeus.common.config import Config


class Module(nn.Cell):
    """Base Module to adapter pytorch Module."""

    data_format = 'channels_first'

    def __init__(self):
        super(Module, self).__init__()
        self.children_ms = []
        self._modules = Config()

    def add_module(self, name, model):
        """Add models into self._models."""
        self.insert_child_to_cell(name, model)
        self.children_ms = list(self._cells.values())

    def children(self):
        """Get child models of current Module."""
        return self.children_ms

    def __setattr__(self, name, value):
        """Overide __setattr__."""
        if isinstance(value, nn.Cell):
            super().__setattr__(name, value)
            self.children_ms = list(self._cells.values())
        else:
            object.__setattr__(self, name, value)

    def named_modules(self):
        """Return names spaces."""
        _names_modules = []
        for model in self.children():
            if isinstance(model, Module) or isinstance(model, nn.Cell):
                _names_modules.append(((model.__class__.__name__, model)))
                if isinstance(model, OperatorSerializable):
                    continue
                child_modules = model.named_modules()
                _names_modules.extend(child_modules)
        return _names_modules

    def initializer(self):
        """Init params."""
        pass

    def call(self, inputs, *args, **kwargs):
        """Call inputs."""
        output = inputs
        models = self.children()
        for model in models:
            if args == ():
                output = model(output)
            else:
                output = model(output, *args)
        return output

    def construct(self, inputs, *args, **kwargs):
        """Construct x."""
        return self.call(inputs, *args, **kwargs)

    def set_parameters(self, name, value):
        """Set Parameters."""
        # self.insert_param_to_cell(name, value)
        setattr(self, name, value)

    def get_weights(self, name):
        """Get Weights."""
        return getattr(self, name)

    def get_weight_ops(self, name):
        """Get weight ops."""
        return self.get_weights(name)


@ClassFactory.register(ClassType.NETWORK)
class QuantizeConv2d(OperatorSerializable):
    """QuantizeConv2d Module inherit nn.Module."""

    def __init__(self):
        """Construct Identity class."""
        OperatorSerializable.__init__(self)

    def construct(self, input):
        """Call QuantizeConv2d function."""
        # todo
        return input


@ClassFactory.register(ClassType.NETWORK)
class AdaptiveAvgPool2d(nn.Cell, OperatorSerializable):
    """Call reduce_mean."""

    def __init__(self, output_size=(1, 1)):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.reduce_mean = P.ReduceMean(keep_dims=True)

    def construct(self, input):
        """Call reduce_mean function."""
        return self.reduce_mean(input, (2, 3))


@ClassFactory.register(ClassType.NETWORK)
class View(nn.Cell, OperatorSerializable):
    """Call squeeze."""

    def __init__(self, size=None):
        super(View, self).__init__()
        self.reshape = P.Reshape()
        self.size = size
        if size is not None and not isinstance(size, tuple):
            self.size = tuple(size)
        self.shape = P.Shape()
        # self.squeeze = P.Squeeze((1, 2))

    def construct(self, inputs):
        """Call squeeze function."""
        if self.size is None:
            return self.reshape(inputs, (self.shape(inputs)[0], -1))
        else:
            return self.reshape(inputs, self.size)
        # return self.squeeze(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Linear(nn.Cell, OperatorSerializable):
    """Call dense."""

    def __init__(self, in_features, out_features, has_bias=True, activation=None):
        super(Linear, self).__init__()
        self.activation = activation
        self.linear = nn.Dense(in_features, out_features, has_bias=has_bias)
        self.linear.update_parameters_name("linear_" + str(np.random.rand()) + ".")

    def construct(self, input):
        """Call dense function."""
        return self.linear(input)


class KaimingNormal(init.Initializer):
    """Call KaimingNormal."""

    def __init__(self, a=0, mode='fan_in', nonlinearity='relu'):
        super(KaimingNormal, self).__init__()
        self.mode = mode
        self.gain = math.sqrt(2.0)

    def _calculate_in_and_out(self, arr):
        dim = len(arr.shape)
        if dim < 2:
            raise ValueError("If initialize data with xavier uniform, the dimension of data must greater than 1.")

        n_in = arr.shape[1]
        n_out = arr.shape[0]

        if dim > 2:
            counter = reduce(lambda x, y: x * y, arr.shape[2:])
            n_in *= counter
            n_out *= counter
        return n_in, n_out

    def _select_fan(self, array, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

        fan_in, fan_out = self._calculate_in_and_out(array)
        return fan_in if mode == 'fan_in' else fan_out

    def _assignment(self, arr, num):
        """Assign the value of `num` to `arr`."""
        if arr.shape == ():
            arr = arr.reshape((1))
            arr[:] = num
            arr = arr.reshape(())
        else:
            if isinstance(num, np.ndarray):
                arr[:] = num[:]
            else:
                arr[:] = num
        return arr

    def _initialize(self, arr):
        fan = self._select_fan(arr, self.mode)
        std = self.gain / math.sqrt(fan)
        np.random.seed(0)
        data = np.random.normal(0, std, arr.shape)

        self._assignment(arr, data)


@ClassFactory.register(ClassType.NETWORK)
class DepthwiseConv2d(nn.Cell, OperatorSerializable):
    """Call DepthwiseConv2d."""

    def __init__(self, in_channels, kernel_size, stride, pad_mode, pad, channel_multiplier=1, has_bias=False,
                 dilation=1):
        super(DepthwiseConv2d, self).__init__()
        self.has_bias = has_bias
        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.out_channels = in_channels * channel_multiplier
        self.kernel_size = (kernel_size, kernel_size)
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier,
                                                      kernel_size=self.kernel_size,
                                                      stride=stride, pad_mode=pad_mode, pad=pad,
                                                      dilation=dilation)
        self.bias_add = P.BiasAdd()
        weight_shape = [channel_multiplier, in_channels, *self.kernel_size]
        self.weight = Parameter(initializer('ones', weight_shape), name='weight')

        if has_bias:
            bias_shape = [channel_multiplier * in_channels]
            self.bias = Parameter(initializer('zeros', bias_shape), name='bias')
        else:
            self.bias = None

    def construct(self, x):
        """Call DepthwiseConv2d function."""
        output = self.depthwise_conv(x, self.weight)
        if self.has_bias:
            output = self.bias_add(output, self.bias)
        return output


@ClassFactory.register(ClassType.NETWORK)
class Conv2d(nn.Cell, OperatorSerializable):
    """Call conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True,
                 groups=1, dilation=1, separable=False, depthwise=False):
        super(Conv2d, self).__init__()
        self.out_channels = out_channels
        if groups == 1:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    has_bias=bias, group=groups, dilation=dilation, pad_mode="pad")
            self.conv2d.update_parameters_name("conv2d_" + str(np.random.rand()) + ".")
        elif in_channels == out_channels and in_channels == groups:
            self.conv2d = DepthwiseConv2d(in_channels, kernel_size=kernel_size, stride=stride, pad_mode="pad",
                                          pad=padding, has_bias=bias, dilation=dilation)
            self.conv2d.update_parameters_name("conv2d_" + str(np.random.rand()) + ".")
        else:
            # TODO delete
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    has_bias=bias, group=1, dilation=dilation, pad_mode="pad")
            self.conv2d.update_parameters_name("conv2d_" + str(np.random.rand()) + ".")
            # raise ValueError("For group not equal to 1, the in_channels, out_chanels and group should be equal.")

    def construct(self, input):
        """Call conv2d function."""
        return self.conv2d(input)

    def initial(self, kernel_mode='he', bias_mode='zero', kernel_scale=1., bias_scale=1.):
        """Initialize weight and bias."""
        if kernel_mode == 'he':
            self.conv2d.weight = init.initializer(  # self.conv2d.weight.default_input for mindspore 0.5~0.7
                KaimingNormal(a=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                self.conv2d.weight.shape, self.conv2d.weight.dtype).to_tensor()
        if bias_mode == "zero":
            self.conv2d.bias = init.initializer(
                'zeros', self.conv2d.bias.shape, self.conv2d.bias.dtype).to_tensor()


@ClassFactory.register(ClassType.NETWORK)
class BatchNorm2d(nn.Cell, OperatorSerializable):
    """Call BatchNorm2d."""

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(BatchNorm2d, self).__init__()

        self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)
        self.batch_norm.update_parameters_name("batchnorm_" + str(np.random.rand()) + ".")

    def construct(self, input):
        """Call batch_norm function."""
        return self.batch_norm(input)


@ClassFactory.register(ClassType.NETWORK)
class SeparableConv2d(nn.Cell, OperatorSerializable):
    """Separable Conv2d  args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_channels, bias=bias, pad_mode="pad")
        self.conv1.update_parameters_name("conv1_" + str(np.random.rand()) + ".")
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.conv2.update_parameters_name("conv2_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call separable_conv2d function."""
        x = self.conv1(x)
        return self.conv2(x)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(nn.Cell, OperatorSerializable):
    """MaxPool2d Module inherit nn.MaxPool2d."""

    def __init__(self, kernel_size, stride, padding=0, pad_mode="valid"):
        super(MaxPool2d, self).__init__()
        self.padding = padding
        if padding > 0:
            self.pad_op = P.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        # if padding != 0:
        #     pad_mode = "same"
        self.max_pool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.max_pool2d.update_parameters_name("maxpool2d_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call maxpool2d function."""
        if self.padding > 0:
            x = self.pad_op(x)
        return self.max_pool2d(x)


@ClassFactory.register(ClassType.NETWORK)
class AvgPool2d(nn.Cell, OperatorSerializable):
    """AvgPool2d Module inherit nn.AvgPool2d."""

    def __init__(self, kernel_size, stride, padding=0, count_include_pad=True, pad_mode="valid"):
        super(AvgPool2d, self).__init__()
        self.padding = padding
        if padding > 0:
            self.pad_op = P.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        # if padding != 0:
        #     pad_mode = "same"
        self.avg_pool2d = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.avg_pool2d.update_parameters_name("avgpool2d_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call maxpool2d function."""
        if self.padding > 0:
            x = self.pad_op(x)
        return self.avg_pool2d(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu(nn.Cell, OperatorSerializable):
    """Relu Module inherit nn.Relu."""

    def __init__(self, inplace=False):
        super(Relu, self).__init__()
        self.relu = nn.ReLU()
        self.relu.update_parameters_name("relu_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call relu function."""
        return self.relu(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(nn.Cell, OperatorSerializable):
    """Relu6 Module inherit nn.Relu6."""

    def __init__(self, inplace=False):
        super(Relu6, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call relu function."""
        return self.relu6(x)


@ClassFactory.register(ClassType.NETWORK)
class Hsigmoid(nn.Cell, OperatorSerializable):
    """Hsigmoid Module."""

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call relu function."""
        return self.relu6(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Hswish(nn.Cell, OperatorSerializable):
    """Hswish Module."""

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + str(np.random.rand()) + ".")

    def construct(self, x):
        """Call relu function."""
        return x * self.relu6(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Identity(nn.Cell, OperatorSerializable):
    """Identity block."""

    def __init__(self):
        """Construct Identity class."""
        super(Identity, self).__init__()

    def construct(self, x):
        """Do an inference on Identity.

        :param x: input tensor
        :return: output tensor
        """
        return x


@ClassFactory.register(ClassType.NETWORK)
class Zero(nn.Cell, OperatorSerializable):
    """Zero block."""

    def __init__(self, stride):
        """Construct Zero class.

        :param stride: stride of the output
        """
        super(Zero, self).__init__()
        self.zeroslike = P.ZerosLike()
        self.stride = stride
        self.shape = P.Shape()

    def construct(self, x):
        """Do an inference on Zero.

        :param x: input tensor
        :return: output tensor
        """
        in_shape = self.shape(x)
        out_shape = (in_shape[0], in_shape[1], in_shape[2] // self.stride, in_shape[3] // self.stride)
        return Tensor(np.zeros(out_shape, np.float32))
        # return self.zeroslike(x)[:, :, ::self.stride, ::self.stride]


def zeros(shape):
    """Create zeros like shape."""
    return Tensor(np.zeros(tuple(shape), np.float32))


@ClassFactory.register(ClassType.NETWORK)
class PixelShuffle(nn.Cell, OperatorSerializable):
    """Class of PixelShuffle."""

    def __init__(self, upscale):
        super(PixelShuffle, self).__init__()
        self.pixel_shuffle = P.DepthToSpace(upscale)

    def construct(self, inputs):
        """Call forward function."""
        return self.pixel_shuffle(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Split(nn.Cell, OperatorSerializable):
    """Class of PixelShuffle."""

    def __init__(self, size=None, dim=0):
        super(Split, self).__init__()
        self.dim = dim
        self.size = size
        self.shape = P.Shape()
        # self.split = P.Split(axis=dim, output_num=size)

    def construct(self, inputs):
        """Call Split function."""
        output_num = self.shape(inputs)[self.dim] // self.size
        split = P.Split(self.dim, output_num)
        return split(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Squeeze(nn.Cell, OperatorSerializable):
    """Class of Squeeze."""

    def __init__(self, dim=0):
        super(Squeeze, self).__init__()
        self.squeee = P.Squeeze(axis=dim)

    def construct(self, inputs):
        """Call Squeeze function."""
        return self.squeee(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Permute(nn.Cell, OperatorSerializable):
    """Class of Permute."""

    def __init__(self, size=None):
        super(Permute, self).__init__()
        self.size = size
        self.permute = P.Transpose()

    def construct(self, inputs):
        """Call Permute function."""
        return self.permute(inputs, tuple(self.size))


@ClassFactory.register(ClassType.NETWORK)
class Stack(nn.Cell, OperatorSerializable):
    """Class of Stack."""

    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim
        self.expand_dim = P.ExpandDims()
        self.concat = P.Concat(axis=dim)

    def construct(self, inputs):
        """Call Stack function."""
        expands = []
        for input in inputs:
            expand = self.expand_dim(input, self.dim)
            expands.append(expand)
        return self.concat(tuple(expands))


@ClassFactory.register(ClassType.NETWORK)
class Transpose(nn.Cell, OperatorSerializable):
    """Class of Transpose."""

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1, self.dim2 = dim1, dim2
        self.transpose = P.Transpose()
        self.shape = P.Shape()

    def construct(self, inputs):
        """Call Transpose function."""
        # new_dim = [i for i in range(len(self.shape(inputs)))]
        # new_dim[self.dim1], new_dim[self.dim2] = new_dim[self.dim2], new_dim[self.dim1]
        # return self.transpose(inputs, tuple(new_dim))
        new_dim = (0, 2, 1, 3, 4)
        return self.transpose(inputs, new_dim)


@ClassFactory.register(ClassType.NETWORK)
class LeakyReLU(nn.Cell, OperatorSerializable):
    """Class of LeakyReLU."""

    def __init__(self, inplace=False, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(alpha=negative_slope)

    def construct(self, inputs):
        """Call forward function."""
        return self.leaky_relu(inputs)


@ClassFactory.register(ClassType.NETWORK)
class InterpolateScale(nn.Cell, OperatorSerializable):
    """Upsample of torch with scale_factor."""

    def __init__(self, scale_factor=None, size=None, mode='bilinear', align_corners=False):
        super(InterpolateScale, self).__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.shape = P.Shape()
        self.mode = mode

    def construct(self, inputs):
        """Call forward function."""
        w, h = self.shape(inputs)[2] * self.scale_factor, self.shape(inputs)[3] * self.scale_factor
        resize = P.ResizeBilinear((w, h), self.align_corners)
        return resize(inputs)


@ClassFactory.register(ClassType.NETWORK)
class MeanShift(nn.Cell, OperatorSerializable):
    """Subtract or add rgb_mean to the image."""

    def __init__(self, rgb_range, rgb_mean, rgb_std=(1.0, 1.0, 1.0), sign=-1):
        """Construct the class MeanShift.

        :param rgb_range: range of tensor, usually 1.0 or 255.0
        :param rgb_mean: mean of rgb value
        :param rgb_std: std of rgb value
        :param sign: -1 for subtract, 1 for add
        """
        super(MeanShift, self).__init__()
        self.conv2d = nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0,
                                has_bias=True, group=1, dilation=1, pad_mode="pad")
        self.conv2d.update_parameters_name("conv2d_" + str(np.random.rand()) + ".")
        std = Tensor(rgb_std, mindspore.float32)
        self.conv2d.weight = Tensor(np.eye(3).reshape(3, 3, 1, 1).astype(np.float32))
        self.reshape = P.Reshape()
        self.div = P.Div()
        self.conv2d.weight = self.div(self.conv2d.weight, self.reshape(std, (3, 1, 1, 1)))
        self.conv2d.bias = sign * rgb_range * Tensor(rgb_mean, mindspore.float32)
        self.conv2d.bias = self.div(self.conv2d.bias, std)
        self.requires_grad = False

    def construct(self, inputs):
        """Call forward function."""
        return self.conv2d(inputs)


@ClassFactory.register(ClassType.NETWORK)
class GlobalMaxPool1d(nn.Cell):
    """Construct the class GlobalMaxPool1d."""

    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()


@ClassFactory.register(ClassType.NETWORK)
class MoudleList(nn.CellList, OperatorSerializable):
    """Create module lists."""

    def __init__(self):
        super(MoudleList, self).__init__()


def concat(inputs, dim=1):
    """Call concat according to backends."""
    return P.Concat(dim)(inputs)
    # if isinstance(inputs, tuple):
    #     return P.Concat(axis=dim)(inputs)
    # elif isinstance(inputs, list):
    #     return P.Concat(axis=dim)(tuple(inputs))
    # else:
    #     raise TypeError("The type of input must be tuple or list, but get {}.".format(type(inputs)))


def mul(a, b):
    """Call mul according to backends."""
    return P.Mul()(a, b)


def random_normal(*size):
    """Apply random values from a normal distribution."""
    # return P.StandardNormal()(size)
    return Tensor(np.random.randn(*size).astype(np.float32))
    # return P.Normal()(size, 0,1)
    # return Parameter(Tensor(np.random.randn(*size)), name="random_" + str(np.random.rand()))


def softmax(input, dim=-1):
    """Apply a softmax function."""
    return nn.Softmax(axis=dim)(input)


def gumbel_softmax(input, dim=-1, tau=1, hard=True, eps=1e-20):
    """Apply a gumbel softmax function."""
    raise NotImplementedError


def to_numpy(input):
    """Apply numpy function."""
    return input.asnumpy()


def mean(input):
    """Apply mean function."""
    return P.ReduceMean(keep_dims=True)(input, (2, 3))


def interpolate(input, size, mode='bilinear', align_corners=False):
    """Apply interpolate function."""
    return P.ResizeBilinear(size=tuple(size), align_corners=align_corners)(input)


def add_n(input):
    """Apply sum function."""
    return sum(input)


def get_shape(input):
    """Get shape."""
    return input.shape


def drop_path(x, prob):
    """Drop path operation.

    :param x: input feature map
    :type x: torch tensor
    :param prob: dropout probability
    :type prob: float
    :return: output feature map after dropout
    :rtype: torch tensor
    """
    # if prob <= 0.:
    #     return x
    # keep = 1. - prob
    #
    # bernoulli_random = P.random.uniform([int(x.get_shape()[0]), 1, 1, 1])
    # mask = P.cast(bernoulli_random < keep, ms.float32)
    # x = P.div(x, keep)
    # x = P.multiply(x, mask)
    return x


def pad(inputs, position):
    """Apply pad function."""
    # TODO the position of torch is a tuple and the order is reversed, but the mindspore is N*2 tuple and is in order
    pad_op = P.Pad(position)
    return pad_op(inputs)


def tensor_abs(inputs):
    """Apply abs function."""
    return P.Abs()(inputs)


def mean_all(inputs):
    """Apply mean_all function."""
    return P.ReduceMean()(inputs)


def maximum(arg1, arg2):
    """Get max item."""
    pass


def minimum(arg1, arg2):
    """Get min item."""
    pass


def new_constant(tensor, size, value, dtype='long'):
    """Return new tensor with shape."""
    pass


def argmax(tensor, dim):
    """Get max and ind from dim."""
    pass


def clamp(x, min=float("-inf"), max=float("inf")):
    """Cet value after clamp."""
    pass


def where(cond):
    """Return index by condition."""
    pass


def unique(inputs):
    """Return the unique elements of the input tensor."""
    pass


def log(inputs):
    """Return the log of the input tensor."""
    pass


def convert_to_tensor(narray, device):
    """Convert numpy to tensor."""
    pass


def new_ones(tensor, size, dtype=None):
    """Return new tensor with shape."""
    pass


def arange(left, right, dtype, device):
    """Rreange from left to right."""
    pass


def compare_where(cond, x, y):
    """Return item by condition."""
    pass


def unsqueeze(inputs, dim):
    """Expand in dim."""
    pass


def expand_as(inputs, tensor):
    """Expand as tensor."""
    pass


def exp(tensor):
    """Return exp(tensor)."""
    pass


def pow(input, exponent, out=None):
    """Calculate the exponent value of the input by element and returns the result tensor."""
    pass


def ones(input_size, out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    pass


def one_hot(inputs, num_classes):
    """Take LongTensor with index values of shape."""
    pass


def to(input, dtype):
    """Convert input to dtype."""
    pass


def reduce_sum(input, dim=0, dtype=None):
    """Apply sum function."""
    pass
