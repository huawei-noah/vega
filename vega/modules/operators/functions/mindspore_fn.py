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

"""Custom functions of mindspore."""

import os
import uuid
import numpy as np
import mindspore.nn as nn
import mindspore
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Parameter, Tensor
from mindspore.common.initializer import initializer, HeNormal
from vega.common.class_factory import ClassType, ClassFactory
from .serializable import OperatorSerializable


class Module(nn.Cell):
    """Base Module to adapter pytorch Module."""

    data_format = 'channels_first'

    def __init__(self):
        super(Module, self).__init__()
        self.children_ms = []
        self._modules = self._cells
        self.need_adjust = True

    def add_module(self, name, model):
        """Add models into self._models."""
        self.insert_child_to_cell(name, model)
        if model:
            model.update_parameters_name(name + '.')
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
            super().__setattr__(name, value)

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

    def _apply_names(self, parent_name=''):
        """Apply names spaces."""
        for scope_name, module in self.name_cells().items():
            scope_name = '{}.{}'.format(parent_name, scope_name) if parent_name else scope_name
            module.name = scope_name + '/' + module.__class__.__name__
            if hasattr(module, "_apply_names"):
                module._apply_names(scope_name)

    def named_children(self):
        """Return names children."""
        return [(name, module) for name, module in self._cells.items()]

    def initialize(self):
        """Init params."""
        pass

    def call(self, *inputs):
        """Call inputs."""
        if len(inputs) == 1:
            output = inputs[0]
            args = ()
        else:
            output = inputs[0]
            args = inputs[1:-1]

        models = self.children()
        for model in models:
            if args == ():
                output = model(output)
            else:
                output = model(output, *args)
        return output

    def construct(self, *inputs):
        """Construct x."""
        return self.call(*inputs)

    def set_parameters(self, name, value):
        """Set Parameters."""
        setattr(self, name, value)
        return 0

    def get_weights(self, name):
        """Get Weights."""
        return getattr(self, name)

    def get_weight_ops(self, name):
        """Get weight ops."""
        return self.get_weights(name)

    def pretrained(self, pretrained_model_file=None):
        """Load pretrained weights."""
        if pretrained_model_file.endswith(".pth"):
            from .pytorch_to_ms import pytorch2mindspore, pytorch2mindspore_extend
            if self.need_adjust:
                ms_pretrained_weight = pytorch2mindspore_extend(pretrained_model_file, self)
            else:
                ms_pretrained_weight = pytorch2mindspore(pretrained_model_file)
        else:
            if os.path.isfile(pretrained_model_file):
                ms_pretrained_weight = pretrained_model_file
            else:
                for file in os.listdir(pretrained_model_file):
                    if file.endswith(".ckpt"):
                        ms_pretrained_weight = os.path.join(pretrained_model_file, file)
                        break
        if self.need_adjust:
            from .adaptive_weight_ms import adaptive_weight
            ms_pretrained_weight = adaptive_weight(ms_pretrained_weight, self)
        return ms_pretrained_weight


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
class Pad(OperatorSerializable, nn.Cell):
    """Pad layer."""

    def __init__(self, output_size=(1, 1)):
        super(Pad, self).__init__()
        self.output_size = output_size

    def construct(self, input):
        """Call Pad function."""
        return input


@ClassFactory.register(ClassType.NETWORK)
class AdaptiveAvgPool2d(OperatorSerializable, nn.Cell):
    """Call reduce_mean."""

    def __init__(self, output_size=(1, 1)):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size
        self.reduce_mean = P.ReduceMean(keep_dims=True)

    def construct(self, input):
        """Call reduce_mean function."""
        return self.reduce_mean(input, (2, 3))


@ClassFactory.register(ClassType.NETWORK)
class View(OperatorSerializable, nn.Cell):
    """Call squeeze."""

    def __init__(self, size=None):
        super(View, self).__init__()
        self.reshape = P.Reshape()
        self.size = size
        if size is not None and not isinstance(size, tuple):
            self.size = tuple(size)
        self.shape = P.Shape()

    def construct(self, inputs):
        """Call squeeze function."""
        if self.size is None:
            return self.reshape(inputs, (self.shape(inputs)[0], -1))
        else:
            return self.reshape(inputs, self.size)


@ClassFactory.register(ClassType.NETWORK)
class Linear(OperatorSerializable, nn.Cell):
    """Call dense."""

    def __init__(self, in_features=None, out_features=None, has_bias=True, activation=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.activation = activation
        self.linear = nn.Dense(in_features, out_features, has_bias=has_bias)
        self.linear.update_parameters_name("linear_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, input):
        """Call dense function."""
        return self.linear(input)


@ClassFactory.register(ClassType.NETWORK)
class DepthwiseConv2d(OperatorSerializable, nn.Cell):
    """Call DepthwiseConv2d."""

    def __init__(self, in_channels, kernel_size, stride, pad_mode, padding, channel_multiplier=1, has_bias=False,
                 dilation=1):
        super(DepthwiseConv2d, self).__init__()
        self.has_bias = has_bias
        self.in_channels = in_channels
        self.channel_multiplier = channel_multiplier
        self.out_channels = in_channels * channel_multiplier
        self.kernel_size = (kernel_size, kernel_size)
        self.depthwise_conv = P.DepthwiseConv2dNative(channel_multiplier=channel_multiplier,
                                                      kernel_size=self.kernel_size,
                                                      stride=stride, pad_mode=pad_mode, pad=padding,
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
class Conv2d(OperatorSerializable, nn.Cell):
    """Call conv2d."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True,
                 groups=1, dilation=1, separable=False, depthwise=False, padding_mode=None):
        super(Conv2d, self).__init__()
        if padding_mode == "same":
            pad_mode = "same"
            padding = 0
        elif padding is None:
            pad_mode = "same"
            padding = 0
        else:
            pad_mode = "pad"
            padding = padding[0] if isinstance(padding, (list, tuple)) else padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = tuple(stride) if isinstance(stride, list) else stride
        dilation = tuple(dilation) if isinstance(dilation, list) else dilation
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                has_bias=bias, group=groups, dilation=dilation, pad_mode=pad_mode)
        self.conv2d.update_parameters_name("conv2d_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, input):
        """Call conv2d function."""
        return self.conv2d(input)

    def initial(self, kernel_mode='he', bias_mode='zero', kernel_scale=1., bias_scale=1.):
        """Initialize weight and bias."""
        return


@ClassFactory.register(ClassType.NETWORK)
class BatchNorm2d(OperatorSerializable, nn.Cell):
    """Call BatchNorm2d."""

    def __init__(self, num_features, eps=1e-05, momentum=0.9, affine=True):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.batch_norm = nn.BatchNorm2d(num_features=num_features, eps=eps, momentum=momentum, affine=affine)
        self.batch_norm.update_parameters_name("batchnorm_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, input):
        """Call batch_norm function."""
        return self.batch_norm(input)


@ClassFactory.register(ClassType.NETWORK)
class SeparableConv2d(OperatorSerializable, nn.Cell):
    """Separable Conv2d  args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_channels, bias=bias, pad_mode="pad")
        self.conv1.update_parameters_name("conv1_" + uuid.uuid1().hex[:8] + ".")
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=bias)
        self.conv2.update_parameters_name("conv2_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call separable_conv2d function."""
        x = self.conv1(x)
        return self.conv2(x)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(OperatorSerializable, nn.Cell):
    """MaxPool2d Module inherit nn.MaxPool2d."""

    def __init__(self, kernel_size, stride, padding=0, pad_mode="valid"):
        super(MaxPool2d, self).__init__()
        self.padding = padding
        if padding > 0:
            self.pad_op = P.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.max_pool2d = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.max_pool2d.update_parameters_name("maxpool2d_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call maxpool2d function."""
        if self.padding > 0:
            x = self.pad_op(x)
        return self.max_pool2d(x)


@ClassFactory.register(ClassType.NETWORK)
class AvgPool2d(OperatorSerializable, nn.Cell):
    """AvgPool2d Module inherit nn.AvgPool2d."""

    def __init__(self, kernel_size, stride, padding=0, count_include_pad=True, pad_mode="valid"):
        super(AvgPool2d, self).__init__()
        self.padding = padding
        if padding > 0:
            self.pad_op = P.Pad(((0, 0), (0, 0), (padding, padding), (padding, padding)))
        self.avg_pool2d = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, pad_mode=pad_mode)
        self.avg_pool2d.update_parameters_name("avgpool2d_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call maxpool2d function."""
        if self.padding > 0:
            x = self.pad_op(x)
        return self.avg_pool2d(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu(OperatorSerializable, nn.Cell):
    """Relu Module inherit nn.Relu."""

    def __init__(self, inplace=False):
        super(Relu, self).__init__()
        self.relu = nn.ReLU()
        self.relu.update_parameters_name("relu_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call relu function."""
        return self.relu(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(OperatorSerializable, nn.Cell):
    """Relu6 Module inherit nn.Relu6."""

    def __init__(self, inplace=False):
        super(Relu6, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call relu function."""
        return self.relu6(x)


@ClassFactory.register(ClassType.NETWORK)
class Hsigmoid(OperatorSerializable, nn.Cell):
    """Hsigmoid Module."""

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call Hsigmoid function."""
        return self.relu6(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Hswish(OperatorSerializable, nn.Cell):
    """Hswish Module."""

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6()
        self.relu6.update_parameters_name("relu6_" + uuid.uuid1().hex[:8] + ".")

    def construct(self, x):
        """Call Hswish function."""
        return x * self.relu6(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Identity(OperatorSerializable, nn.Cell):
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
class Dropout(OperatorSerializable, nn.Cell):
    """Dropout block."""

    def __init__(self, prob=0.5):
        """Construct Dropout class."""
        super(Dropout, self).__init__()
        if prob == 0:
            prob = 1e-12
        self.dropout = nn.Dropout(1 - prob)

    def construct(self, x, **kwargs):
        """Do an inference on Dropout."""
        return self.dropout(x)


@ClassFactory.register(ClassType.NETWORK)
class Zero(OperatorSerializable, nn.Cell):
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
        return self.zeroslike(x[:, :, ::self.stride, ::self.stride])


def zeros(shape):
    """Create zeros like shape."""
    return Tensor(np.zeros(tuple(shape), np.float32))


@ClassFactory.register(ClassType.NETWORK)
class PixelShuffle(OperatorSerializable, nn.Cell):
    """Class of PixelShuffle."""

    def __init__(self, upscale):
        super(PixelShuffle, self).__init__()
        self.pixel_shuffle = P.DepthToSpace(upscale)

    def construct(self, inputs):
        """Call forward function."""
        return self.pixel_shuffle(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Split(OperatorSerializable, nn.Cell):
    """Class of PixelShuffle."""

    def __init__(self, size=None, dim=0):
        super(Split, self).__init__()
        self.dim = dim
        self.size = size
        self.shape = P.Shape()

    def construct(self, inputs):
        """Call Split function."""
        output_num = self.shape(inputs)[self.dim] // self.size
        split = P.Split(self.dim, output_num)
        return split(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Squeeze(OperatorSerializable, nn.Cell):
    """Class of Squeeze."""

    def __init__(self, dim=0):
        super(Squeeze, self).__init__()
        self.squeee = P.Squeeze(axis=dim)

    def construct(self, inputs):
        """Call Squeeze function."""
        return self.squeee(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Permute(OperatorSerializable, nn.Cell):
    """Class of Permute."""

    def __init__(self, size=None):
        super(Permute, self).__init__()
        self.size = size
        self.permute = P.Transpose()

    def construct(self, inputs):
        """Call Permute function."""
        return self.permute(inputs, self.size)


@ClassFactory.register(ClassType.NETWORK)
class Stack(OperatorSerializable, nn.Cell):
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
            expand_input = self.expand_dim(input, self.dim)
            expands.append(expand_input)
        return self.concat(tuple(expands))


@ClassFactory.register(ClassType.NETWORK)
class Transpose(OperatorSerializable, nn.Cell):
    """Class of Transpose."""

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1, self.dim2 = dim1, dim2
        self.transpose = P.Transpose()
        self.shape = P.Shape()

    def construct(self, inputs):
        """Call Transpose function."""
        new_dim = ()
        for i in range(len(self.shape(inputs))):
            if i == self.dim1:
                index = self.dim2
            elif i == self.dim2:
                index = self.dim1
            else:
                index = i
            new_dim = new_dim + (index,)
        return self.transpose(inputs, new_dim)


@ClassFactory.register(ClassType.NETWORK)
class LeakyReLU(OperatorSerializable, nn.Cell):
    """Class of LeakyReLU."""

    def __init__(self, inplace=False, negative_slope=0.01):
        super(LeakyReLU, self).__init__()
        self.leaky_relu = nn.LeakyReLU(alpha=negative_slope)

    def construct(self, inputs):
        """Call forward function."""
        return self.leaky_relu(inputs)


@ClassFactory.register(ClassType.NETWORK)
class InterpolateScale(OperatorSerializable, nn.Cell):
    """Upsample of torch with scale_factor."""

    def __init__(self, scale_factor=None, size=None, mode='bilinear', align_corners=False):
        super(InterpolateScale, self).__init__()
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        self.shape = P.Shape()
        self.mode = mode
        self.size = size

    def construct(self, inputs):
        """Call forward function."""
        if self.size is not None:
            resize = P.ResizeBilinear(self.size, self.align_corners)
        else:
            w, h = self.shape(inputs)[2] * self.scale_factor, self.shape(inputs)[3] * self.scale_factor
            resize = P.ResizeBilinear((w, h), self.align_corners)

        return resize(inputs)


@ClassFactory.register(ClassType.NETWORK)
class MeanShift(OperatorSerializable, nn.Cell):
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
        self.conv2d.update_parameters_name("conv2d_" + uuid.uuid1().hex[:8] + ".")
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


@ClassFactory.register(ClassType.NETWORK)
class Tanh(nn.Tanh, OperatorSerializable):
    """Class of Dropout."""

    pass


@ClassFactory.register(ClassType.NETWORK)
class Embedding(nn.Embedding, OperatorSerializable):
    """Class of Dropout."""

    pass


def concat(inputs, dim=1):
    """Call concat according to backends."""
    return P.Concat(dim)(inputs)


def mul(a, b):
    """Call mul according to backends."""
    return P.Mul()(a, b)


def random_normal(*size):
    """Apply random values from a normal distribution."""
    return Tensor(np.random.randn(*size).astype(np.float32))


def softmax(input, dim=-1):
    """Apply a softmax function."""
    return nn.Softmax(dim)(input)


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
    return P.AddN()(input)


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
    return x


def pad(inputs, position):
    """Apply pad function."""
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


def ones_like(out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    pass


def zeros_like(out):
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


def gelu(x):
    """Apply gelu function."""
    return nn.GELU()(x)


def swish(x):
    """Apply swish function."""
    pass


def relu(x):
    """Apply relu function."""
    pass


def sqrt(x):
    """Apply sqrt function."""
    pass


def matmul(x1, x2):
    """Apply matmul function."""
    return ops.matmul(x1, x2)


@ClassFactory.register(ClassType.NETWORK)
class LayerNorm(OperatorSerializable, nn.Cell):
    """Layer Norm module."""

    def __init__(self, in_channels=None, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm((in_channels,))

    def construct(self, input_tensor):
        """Layer norm."""
        return self.layer_norm(input_tensor)


@ClassFactory.register(ClassType.NETWORK)
class Flatten(OperatorSerializable, nn.Cell):
    """Flatten module."""

    def __init__(self, start_dim=0):
        super(Flatten, self).__init__()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.start_dim = start_dim

    def construct(self, x):
        """Apply Flatten."""
        old_shape = self.shape(x)
        flatten_dim = old_shape[self.start_dim:]
        flatten_size = 1
        for i in range(len(flatten_dim)):
            flatten_size = flatten_size * flatten_dim[i]
        new_shape = old_shape[0:self.start_dim] + (flatten_size,)
        return self.reshape(x, new_shape)


def expand(x, expand_shape):
    """Expand a tensor."""
    return P.Tile()(x, expand_shape)


def MSELoss():
    """MSE Loss."""
    pass
