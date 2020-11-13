# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Custom functions of pytorch."""
import torch
import torch.nn as nn
from torch.functional import F
import torch.nn.init as init
from .serializable import OperatorSerializable
from torch.nn.quantized import Conv2d as QuantConv2d
from zeus.common.class_factory import ClassType, ClassFactory


class Module(nn.Module):
    """Base Module to adapter pytorch Module."""

    data_format = 'channels_first'

    def __init__(self):
        super(Module, self).__init__()
        self._is_cuda = False

    def initializer(self):
        """Init params."""
        pass

    def set_parameters(self, name, value):
        """Set Parameters."""
        self.register_buffer(name, value.cuda().requires_grad_())

    def get_weights(self, name):
        """Get Weights."""
        return getattr(self, name)

    def get_weight_ops(self, name):
        """Get weight ops."""
        return self.get_weights(name)

    def call(self, inputs, *args, **kwargs):
        """Call inputs."""
        output = inputs
        models = self.children()
        for model in models:
            output = model(output)
        return output

    def cuda(self, device=None):
        """Set cuda flag."""
        self._is_cuda = True
        return super().cuda(device)

    @property
    def is_cuda(self):
        """Judge is cuda."""
        return self._is_cuda

    def forward(self, inputs, *args, **kwargs):
        """Call compiled function."""
        return self.call(inputs, *args, **kwargs)


@ClassFactory.register(ClassType.NETWORK)
class QuantizeConv2d(QuantConv2d, Module, OperatorSerializable):
    """QuantizeConv2d Module inherit nn.Module."""

    _quant_type = {8: torch.quint8}

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', quant_bit=None):
        """Construct Identity class."""
        self.quant_bit = quant_bit
        OperatorSerializable.__init__(self)
        Module.__init__(self)
        QuantConv2d.__init__(self, in_channels=in_channels, out_channels=out_channels,
                             kernel_size=kernel_size, stride=stride, padding=padding,
                             dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

    def forward(self, input):
        """Do an inference on Identity."""
        input = input.cpu()
        input = torch.quantize_per_tensor(input, 1.0, 0, self._quant_type[self.quant_bit])
        output = super().forward(input)
        output = torch.dequantize(output).cuda()
        return output


@ClassFactory.register(ClassType.NETWORK)
class Conv2d(nn.Conv2d, OperatorSerializable):
    """Conv2d Module inherit nn.Module."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, groups=1,
                 dilation=1, separable=False, depthwise=False):
        """Construct Identity class."""
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, groups=groups,
                                     padding=padding, bias=bias, dilation=dilation)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)

    def initial(self, kernel_mode='he', bias_mode='zero', kernel_scale=1., bias_scale=1.):
        """Initialize weight and bias."""
        if kernel_mode == 'he':
            init.kaiming_normal_(self.weight, a=0, mode='fan_in')
            self.weight.data *= kernel_scale
        if bias_mode == 'zero':
            if self.bias is not None:
                self.bias.data.zero_()


@ClassFactory.register(ClassType.NETWORK)
class SeparableConv2d(nn.Module, OperatorSerializable):
    """Separable Conv2d  args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.conv2 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        """Call separable_conv2d function."""
        return self.conv2(self.conv1(x))


@ClassFactory.register(ClassType.NETWORK)
class BatchNorm2d(nn.BatchNorm2d, OperatorSerializable):
    """BatchNorm2d Module inherit nn.BatchNorm2d."""

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True):
        """Construct Identity class."""
        super(BatchNorm2d, self).__init__(num_features, eps, momentum, affine)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(nn.MaxPool2d, OperatorSerializable):
    """MaxPool2d Module inherit nn.MaxPool2d."""

    def __init__(self, kernel_size, stride=1, padding=0):
        """Construct MaxPool2d class."""
        super(MaxPool2d, self).__init__(kernel_size, stride, padding)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class AvgPool2d(nn.AvgPool2d, OperatorSerializable):
    """AvgPool2d Module inherit nn.AvgPool2d."""

    def __init__(self, kernel_size, stride=1, padding=0, count_include_pad=True):
        """Construct Identity class."""
        if not stride:
            stride = kernel_size
        super(AvgPool2d, self).__init__(kernel_size, stride,
                                        padding, count_include_pad=count_include_pad)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu(nn.ReLU, OperatorSerializable):
    """Relu Module inherit nn.Relu."""

    def __init__(self, inplace=False):
        """Construct ReLU class."""
        super(Relu, self).__init__(inplace)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(nn.ReLU6, OperatorSerializable):
    """Relu6 Module inherit nn.Relu6."""

    def __init__(self, inplace=False):
        """Construct ReLU class."""
        super(Relu6, self).__init__(inplace)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Hswish(nn.ReLU6, OperatorSerializable):
    """Call Hswish."""

    def __init__(self, inplace=False):
        super(Hswish, self).__init__(inplace)

    def __call__(self, x, **kwargs):
        """Call Hswish function."""
        return x * super().forward(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Hsigmoid(nn.ReLU6, OperatorSerializable):
    """Call Hsigmoid."""

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__(inplace)
        self.inplace = inplace

    def __call__(self, x, **kwargs):
        """Call Hsigmoid function."""
        return super().forward(x + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, OperatorSerializable):
    """AdaptiveAvgPool2d Module inherit nn.AdaptiveAvgPool2d."""

    def __init__(self, output_size=(1, 1)):
        """Construct Linear class."""
        super(AdaptiveAvgPool2d, self).__init__(output_size)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Linear(nn.Linear, OperatorSerializable):
    """Linear Module inherit nn.Linear."""

    def __init__(self, in_features, out_features, use_bias=True, activation=None):
        """Construct Linear class."""
        super(Linear, self).__init__(in_features, out_features, use_bias)
        self.activation = activation

    def forward(self, x):
        """Do an inference on Identity."""
        out = super().forward(x)
        if self.activation == 'softmax':
            return F.softmax(out)
        return out


@ClassFactory.register(ClassType.NETWORK)
class Identity(nn.Module, OperatorSerializable):
    """Identity block."""

    def __init__(self):
        """Construct Identity class."""
        super(Identity, self).__init__()

    def forward(self, x):
        """Do an inference on Identity.

        :param x: input tensor
        :return: output tensor
        """
        return x


@ClassFactory.register(ClassType.NETWORK)
class Zero(nn.Module, OperatorSerializable):
    """Zero block."""

    def __init__(self, stride):
        """Construct Zero class.

        :param stride: stride of the output
        """
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        """Do an inference on Zero.

        :param x: input tensor
        :return: output tensor
        """
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


@ClassFactory.register(ClassType.NETWORK)
class View(nn.Module, OperatorSerializable):
    """View Function of torch."""

    def __init__(self, size=None):
        self.size = size
        super(View, self).__init__()

    def forward(self, inputs):
        """Call forward function."""
        if not self.size:
            return inputs.view(inputs.size(0), -1)
        else:
            return inputs.view(*self.size)


@ClassFactory.register(ClassType.NETWORK)
class PixelShuffle(nn.PixelShuffle, OperatorSerializable):
    """PixelShuffle of torch."""

    def __init__(self, upscale):
        super(PixelShuffle, self).__init__(upscale)

    def forward(self, inputs):
        """Call forward function."""
        return super().forward(inputs)


@ClassFactory.register(ClassType.NETWORK)
class Split(nn.Module, OperatorSerializable):
    """Split of torch."""

    def __init__(self, size=None, dim=0):
        super(Split, self).__init__()
        self.size = size
        self.dim = dim

    def forward(self, inputs):
        """Call forward function."""
        return torch.split(inputs, self.size, self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Squeeze(nn.Module, OperatorSerializable):
    """Squeeze of torch."""

    def __init__(self, dim=0):
        self.dim = dim
        super(Squeeze, self).__init__()

    def forward(self, inputs):
        """Call forward function."""
        # return torch.squeeze(inputs, self.dim)
        return inputs.squeeze(self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Permute(nn.Module, OperatorSerializable):
    """Permute of torch."""

    def __init__(self, size=None):
        super(Permute, self).__init__()
        self.size = size

    def forward(self, inputs):
        """Call forward function."""
        return inputs.permute(*self.size).contiguous()


@ClassFactory.register(ClassType.NETWORK)
class Stack(nn.Module, OperatorSerializable):
    """Stack of torch."""

    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        """Call forward function."""
        return torch.stack(inputs, self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Transpose(nn.Module, OperatorSerializable):
    """Class of Transpose."""

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, inputs):
        """Forward function of Transpose."""
        return torch.transpose(inputs, self.dim1, self.dim2).contiguous()


@ClassFactory.register(ClassType.NETWORK)
class LeakyReLU(nn.LeakyReLU, OperatorSerializable):
    """Relu Module inherit nn.LeakyReLU."""

    def __init__(self, inplace=False, negative_slope=0.01):
        """Construct ReLU class."""
        super(LeakyReLU, self).__init__(
            negative_slope=negative_slope, inplace=inplace)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class InterpolateScale(nn.Upsample, OperatorSerializable):
    """Upsample of torch with scale_factor."""

    def __init__(self, scale_factor=None, size=None, mode='bilinear', align_corners=None):
        super(InterpolateScale, self).__init__(scale_factor=scale_factor, size=size, mode=mode,
                                               align_corners=align_corners)

    def forward(self, inputs):
        """Call forward function."""
        return super().forward(inputs)


@ClassFactory.register(ClassType.NETWORK)
class MeanShift(nn.Conv2d, OperatorSerializable):
    """Subtract or add rgb_mean to the image."""

    def __init__(self, rgb_range, rgb_mean, rgb_std=(1.0, 1.0, 1.0), sign=-1):
        """Construct the class MeanShift.

        :param rgb_range: range of tensor, usually 1.0 or 255.0
        :param rgb_mean: mean of rgb value
        :param rgb_std: std of rgb value
        :param sign: -1 for subtract, 1 for add
        """
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False


@ClassFactory.register(ClassType.NETWORK)
class GlobalMaxPool1d(nn.Module):
    """Construct the class GlobalMaxPool1d."""

    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        """Call max_pool1d function."""
        return F.max_pool1d(x, kernel_size=x.shape[2])


@ClassFactory.register(ClassType.NETWORK)
class MoudleList(nn.ModuleList, OperatorSerializable):
    """Class of LeakyReLU."""

    def __init__(self):
        super(MoudleList, self).__init__()


def concat(inputs, dim=1):
    """Call concat according to backends."""
    return torch.cat(inputs, dim=dim)


def mul(a, b):
    """Call mul according to backends."""
    return torch.mul(a, b)


def random_normal(*size):
    """Apply random values from a normal distribution."""
    return torch.randn(*size)


def softmax(input, dim=None):
    """Apply a softmax function."""
    return F.softmax(input, dim)


def gumbel_softmax(input, dim=-1, tau=1, hard=True, eps=1e-20):
    """Apply a softmax function."""
    return F.gumbel_softmax(input, tau, hard, eps, dim)


def to_numpy(input):
    """Apply numpy function."""
    return input.data.cpu().numpy()


def mean(inputs):
    """Apply mean function."""
    return inputs.mean(2, keepdim=True).mean(3, keepdim=True)


def tensor_abs(inputs):
    """Apply abs function."""
    return torch.abs(inputs)


def mean_all(inputs):
    """Apply mean_all function."""
    return torch.mean(inputs)


def pad(inputs, position):
    """Apply pad function."""
    return F.pad(inputs, position)


def interpolate(input, size, mode='bilinear', align_corners=False):
    """Apply interpolate function."""
    return nn.functional.interpolate(input, size=size, mode=mode, align_corners=align_corners)


def add_n(input):
    """Apply sum function."""
    return sum(input)


def get_shape(input):
    """Get shape."""
    return input.size()


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
    mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep)
    x.div_(keep)
    x.mul_(mask)
    return x


def zeros(shape):
    """Create zeros like shape."""
    return torch.zeros(shape).cuda()


def moduleList():
    """Create module lists."""
    return nn.ModuleList()


def maximum(arg1, arg2):
    """Get max item."""
    return torch.max(arg1, arg2)


def minimum(arg1, arg2):
    """Get min item."""
    return torch.min(arg1, arg2)


def new_constant(tensor, size, value, dtype='long'):
    """Return new tensor with shape."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'int':
        dtype = torch.uint8
    else:
        dtype = None
    if isinstance(size, int):
        size = (size,)
    if dtype is None:
        return tensor.new_full(size, value)
    else:
        return tensor.new_full(size, value, dtype=dtype)


def argmax(tensor, dim):
    """Get max and ind from dim."""
    return torch.max(tensor, dim)


def clamp(x, min=float("-inf"), max=float("inf")):
    """Cet value after clamp."""
    return torch.clamp(x, min=min, max=max)


def where(cond):
    """Return index by condition."""
    return torch.nonzero(cond)


def unique(inputs):
    """Return the unique elements of the input tensor."""
    return torch.unique(inputs)


def log(inputs):
    """Return the log of the input tensor."""
    return torch.log(inputs)


def convert_to_tensor(narray, device):
    """Convert numpy to tensor."""
    return torch.from_numpy(narray).long().to(device)


def new_ones(tensor, size, dtype=None):
    """Return new tensor with shape."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'uint8':
        dtype = torch.uint8
    else:
        dtype = None
    if dtype is None:
        return tensor.new_ones(size)
    else:
        return tensor.new_ones(size, dtype=dtype)


def arange(left, right, dtype, device):
    """Reange from left to right."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'uint8':
        dtype = torch.uint8
    else:
        dtype = None
    return torch.arange(left, right, dtype=dtype, device=device)


def compare_where(cond, x, y):
    """Return item by condition."""
    return torch.where(cond, x, y)


def unsqueeze(inputs, dim):
    """Expand in dim."""
    return inputs.unsqueeze(dim)


def expand_as(inputs, tensor):
    """Expand as tensor."""
    return inputs.expand_as(tensor)


def exp(tensor):
    """Return exp(tensor)."""
    return tensor.exp()


def pow(input, exponent, out=None):
    """Calculate the exponent value of the input by element and returns the result tensor."""
    return torch.pow(input, exponent, out=out)


def ones(input_size, out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return torch.ones(input_size, out)


def one_hot(inputs, num_classes, dtype=None):
    """Take LongTensor with index values of shape."""
    return F.one_hot(inputs, num_classes)


def to(input, dtype):
    """Convert input to dtype."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'uint8':
        dtype = torch.uint8
    elif dtype == 'float32':
        dtype = torch.float32
    return input.to(dtype)


def reduce_sum(input, dim=0, dtype=None):
    """Apply sum function."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'uint8':
        dtype = torch.uint8
    elif dtype == 'float32':
        dtype = torch.float32
    return torch.sum(input, dim=dim, dtype=dtype)
