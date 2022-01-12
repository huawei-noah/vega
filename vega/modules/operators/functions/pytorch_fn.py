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

"""Custom functions of pytorch."""
from collections import OrderedDict
import logging
import math
import torch
import torch.nn as nn
from torch.functional import F
import torch.nn.init as init
from torch.nn.quantized import Conv2d as QuantConv2d
from torch.nn import Parameter as Torch_Parameter
import vega
from vega.common.class_factory import ClassType, ClassFactory
from .serializable import OperatorSerializable


class Module(nn.Module):
    """Base Module to adapter pytorch Module."""

    def __init__(self):
        super(Module, self).__init__()
        self._is_cuda = False
        self.strict = True
        self.data_format = 'channels_first'
        self.exclude_weight_prefix = None
        self.weight_file = None
        self._apply_once = True
        self._arch_type = None
        self._before_call_hooks = OrderedDict()

    def build(self):
        """Build network or params."""
        pass

    @classmethod
    def remap_state_dict(self, own_state_dict, state_dict, head_prefix=None):
        """Remap state dict from npu state files."""
        if "state_dict" in state_dict.keys():
            state_dict = state_dict["state_dict"]
        own_keys = [k for k in own_state_dict.keys() if not k.startswith(head_prefix)] if head_prefix else list(
            own_state_dict.keys())
        input_keys = list(state_dict.keys())
        new_state_dict = {}
        own_key_prefix_occurrence_map = {}
        input_key_prefix_occurrence_map = {}

        def _has_prefix(key, prefixes):
            if not prefixes or not key:
                return False
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            for prefix in prefixes:
                if key.startswith(prefix):
                    return True
            return False

        for i in range(len(own_keys)):
            own_key = own_keys[i]
            input_key = input_keys[i]
            if _has_prefix(input_key, head_prefix):
                continue

            own_key_prefix = own_key[:own_key.rfind(".")] if own_key.rfind(".") != -1 else own_key
            own_key_suffix = own_key[own_key.rfind("."):] if own_key.rfind(".") != -1 else own_key
            input_key_prefix = input_key[:input_key.rfind(".")] if input_key.rfind(".") != -1 else input_key
            input_key_suffix = input_key[input_key.rfind("."):] if input_key.rfind(".") != -1 else input_key
            if own_key_prefix not in own_key_prefix_occurrence_map.keys():
                own_key_prefix_occurrence_map[own_key_prefix] = \
                    sum(s.startswith(own_key_prefix + ".") for s in own_keys)
            if input_key_prefix not in input_key_prefix_occurrence_map.keys():
                input_key_prefix_occurrence_map[input_key_prefix] = \
                    sum(s.startswith(input_key_prefix + ".") for s in input_keys)
            own_key_prefix_occurrence = own_key_prefix_occurrence_map[own_key_prefix]
            input_key_prefix_occurrence = input_key_prefix_occurrence_map[input_key_prefix]
            if own_key_prefix_occurrence == input_key_prefix_occurrence and own_key_suffix == input_key_suffix:
                new_state_dict[own_key] = state_dict[input_key]
            else:
                raise Exception("unmatched own_key {} and input_key {}".format(own_key, input_key))

        return new_state_dict

    def load_state_dict(self, state_dict=None, strict=None, file_path=None,
                        exclude_weight_prefix=None):
        """Load state dict from state_dict or file."""
        state_dict = torch.load(file_path) if file_path is not None else state_dict
        self.strict = strict if strict is not None else self.strict
        state_dict = self._exclude_checkpoint_by_prefix(state_dict)
        own_states = self.state_dict()
        if vega.is_npu_device():
            state_dict = self.remap_state_dict(own_states, state_dict, exclude_weight_prefix)
        not_swap_keys = []
        for own_key, own_state in own_states.items():
            state = state_dict.get(own_key)
            if state is None or own_state.shape != state.shape:
                if 'num_batches_tracked' in own_key:
                    continue
                not_swap_keys.append(own_key)
            else:
                own_states[own_key] = state
        if not_swap_keys:
            logging.info("Not Swap Keys: {}".format(not_swap_keys))
        super(Module, self).load_state_dict(state_dict, self.strict)
        return not_swap_keys

    def freeze(self, parameter_to_train=None):
        """Freeze parameters."""
        for name, parameter in self.named_parameters():
            if parameter_to_train:
                if all([not name.startswith(layer) for layer in parameter_to_train]):
                    parameter.requires_grad_(False)
                else:
                    parameter.requires_grad_(False)
        for name, module in self.named_modules():
            if isinstance(module, BatchNorm2d) or isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _exclude_checkpoint_by_prefix(self, states):
        if self.exclude_weight_prefix:
            if not isinstance(self.exclude_weight_prefix, list):
                self.exclude_weight_prefix = [self.exclude_weight_prefix]
            for prefix in self.exclude_weight_prefix:
                states = {k: v for k, v in states.items() if not k.startswith(prefix)}
            self.strict = False
        return states

    def set_parameters(self, name, value):
        """Set Parameters."""
        if vega.is_npu_device():
            self.register_parameter(name, nn.Parameter(value.to(vega.get_devices())))
        elif vega.is_gpu_device():
            self.register_parameter(name, nn.Parameter(value.cuda()))
        else:
            self.register_parameter(name, nn.Parameter(value))
        return getattr(self, name)

    def get_weights(self, name):
        """Get Weights."""
        return getattr(self, name)

    def get_weight_ops(self, name):
        """Get weight ops."""
        return self.get_weights(name)

    def call(self, inputs=None, *args, **kwargs):
        """Call inputs."""
        output = inputs
        models = self.children()
        for model in models:
            output = model(output, *args, **kwargs)
        return output

    def _apply_names(self):
        """Apply names spaces."""
        for name, module in self.named_modules():
            if not hasattr(module, 'name') or not module.name:
                module.name = name

    def forward(self, inputs=None, *args, **kwargs):
        """Call compiled function."""
        if self._apply_once:
            self.build()
            if self.weight_file is not None:
                logging.info("Start to load weight file : {}".format(self.weight_file))
                self.load_state_dict(torch.load(self.weight_file))
            self._apply_once = False
        if inputs is None and kwargs:
            return self.call(**kwargs)
        return self.call(inputs, *args, **kwargs)


@ClassFactory.register(ClassType.NETWORK)
class Pad(nn.Module, OperatorSerializable):
    """Pad block."""

    def __init__(self, mode="constant", padding=None):
        self.mode = mode
        self.padding = padding
        super().__init__()

    def forward(self, input, pads=None, value=0):
        """Call forward."""
        if self.padding is not None:
            pads = self.padding
        elif pads is None:
            raise TypeError("forward() missing 1 required positional argument: 'pads'")
        return F.pad(input, list(pads), mode=self.mode, value=value)


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
        if vega.is_npu_device():
            output = torch.dequantize(output).to(vega.get_devices())
        elif vega.is_gpu_device():
            output = torch.dequantize(output).cuda()
        else:
            output = torch.dequantize(output)
        return output


@ClassFactory.register(ClassType.NETWORK)
class ConvTranspose2d(nn.ConvTranspose2d, OperatorSerializable):
    """MaxPool2d Module inherit nn.MaxPool2d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, output_padding=0, groups=1, bias=True,
                 dilation=1, padding_mode='zeros'):
        """Construct MaxPool2d class."""
        if isinstance(padding, str):
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [v // 2 for v in kernel_size]
        padding = padding if not isinstance(padding, str) else kernel_size // 2
        super(ConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                              groups=groups, padding=padding, bias=bias, dilation=dilation)

    def forward(self, x, output_size=None):
        """Do an inference on Identity."""
        return super().forward(x, output_size)


@ClassFactory.register(ClassType.NETWORK)
class Conv2d(nn.Conv2d, OperatorSerializable):
    """Conv2d Module inherit nn.Module."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False, groups=1,
                 dilation=1, separable=False, depthwise=False, padding_mode="same"):
        """Construct Identity class."""
        if isinstance(padding, str):
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [v // 2 for v in kernel_size]
        padding = padding if not isinstance(padding, str) else kernel_size // 2
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

    @classmethod
    def from_module(cls, module):
        """Convert nn.Conv2d to ops.Conv2d."""
        in_channels = module.in_channels
        out_channels = module.out_channels
        kernel_size = module.kernel_size
        stride = module.stride
        bias = True if module.bias else False
        padding = module.padding
        groups = module.groups
        dilation = module.dilation
        return cls(in_channels, out_channels, kernel_size, stride, padding, bias, groups, dilation)

    def get_weights(self):
        """Get weights."""
        return self._parameters

    def set_weights(self, name, weight):
        """Set weights."""
        if name == 'weight':
            self.weight.data = weight
        elif name == 'bias':
            self.bias.data = weight


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

    def get_weights(self):
        """Get weights."""
        weights = {**self._parameters, **self._buffers}
        if 'num_batches_tracked' in weights:
            weights.pop('num_batches_tracked')
        return weights

    def set_weights(self, name, weight):
        """Set weights."""
        if name == 'running_mean':
            self.running_mean.data = weight
        elif name == 'running_var':
            self.running_var.data = weight
        elif name == 'weight':
            self.weight.data = weight
        elif name == 'bias':
            self.bias.data = weight


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(nn.MaxPool2d, OperatorSerializable):
    """MaxPool2d Module inherit nn.MaxPool2d."""

    def __init__(self, kernel_size, stride=1, padding=0):
        """Construct MaxPool2d class."""
        padding = padding if not isinstance(padding, str) else 1
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
        super(Relu, self).__init__(inplace=False)

    def forward(self, x):
        """Do an inference on Relu."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(nn.ReLU6, OperatorSerializable):
    """Relu6 Module inherit nn.Relu6."""

    def __init__(self, inplace=False):
        """Construct Relu6 class."""
        super(Relu6, self).__init__(inplace)

    def forward(self, x):
        """Do an inference on Relu6."""
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
        """Do an inference on AdaptiveAvgPool2d."""
        if isinstance(x, list):
            x = x[0]
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Linear(nn.Linear, OperatorSerializable):
    """Linear Module inherit nn.Linear."""

    def __init__(self, in_features, out_features, use_bias=True, activation=None):
        """Construct Linear class."""
        super(Linear, self).__init__(in_features, out_features, use_bias)
        self.activation = activation

    def forward(self, x):
        """Do an inference on Linear."""
        out = super().forward(x)
        if self.activation == 'softmax':
            return F.softmax(out)
        return out

    def get_weights(self):
        """Get weights."""
        return self._parameters

    def set_weights(self, name, weight):
        """Set weights."""
        if name == 'weight':
            self.weight.data = weight
        elif name == 'bias':
            self.bias.data = weight


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
class Dropout(nn.Dropout, OperatorSerializable):
    """Dropout Module inherit nn.Dropout."""

    def __init__(self, prob=0.5, inplace=False):
        """Construct Dropout class."""
        super(Dropout, self).__init__(prob, inplace)

    def forward(self, x):
        """Do an inference on Dropout."""
        out = super().forward(x)
        return out


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


@ClassFactory.register(ClassType.NETWORK)
class Tanh(nn.Tanh, OperatorSerializable):
    """Class of Dropout."""

    def forward(self, x):
        """Forward Tanh."""
        return super(Tanh, self).forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Embedding(nn.Embedding, OperatorSerializable):
    """Class of Dropout."""

    def forward(self, x):
        """Call embedding."""
        return super(Embedding, self).forward(x)


def concat(inputs, dim=1):
    """Call concat according to backends."""
    return torch.cat(inputs, dim=dim)


def mul(a, b):
    """Call mul according to backends."""
    return torch.mul(a, b)


def cat(a, b):
    """Call mul according to backends."""
    return torch.mul(a, b)


def matmul(a, b):
    """Call matmul according to backends."""
    return torch.matmul(a, b)


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
    dtype = inputs.dtype
    return F.pad(inputs.cpu().float(), position).to(vega.get_devices()).to(dtype)


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
    if vega.is_gpu_device():
        mask = torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep)
    elif vega.is_npu_device():
        mask = torch.npu.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep)
    x.div_(keep)
    x.mul_(mask)
    return x


def zeros(shape):
    """Create zeros like shape."""
    return torch.zeros(shape)


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


def where(cond, x, y):
    """Return index by condition."""
    return torch.where(cond, x, y)


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


def arange(*inputs, dtype='long', device=None):
    """Rearange from left to right."""
    if dtype == 'long':
        dtype = torch.long
    elif dtype == 'uint8':
        dtype = torch.uint8
    else:
        dtype = None
    return torch.arange(*inputs, dtype=dtype, device=device)


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


def ones(input_size, out=None):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return torch.ones(input_size, out=out)


def ones_like(out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return torch.ones_like(out)


def zeros_like(out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return torch.zeros_like(out)


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


def gelu(x):
    """Apply gelu function."""
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    """Apply swish function."""
    return x * torch.sigmoid(x)


def relu(x, inplace=False):
    """Apply relu function."""
    return F.relu(x, inplace)


def sqrt(x):
    """Apply sqrt function."""
    return torch.sqrt(x)


@ClassFactory.register(ClassType.NETWORK)
class GroupNorm(nn.GroupNorm, OperatorSerializable):
    """GroupNorm Module inherit nn.GroupNorm."""

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        """Construct Identity class."""
        super(GroupNorm, self).__init__(num_groups, num_channels, eps, affine)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


@ClassFactory.register(ClassType.NETWORK)
class SyncBatchNorm(nn.SyncBatchNorm, OperatorSerializable):
    """SyncBatchNorm Module inherit nn.SyncBatchNorm."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, process_group=None):
        """Construct Identity class."""
        super(SyncBatchNorm, self).__init__(num_features, eps, momentum, affine, track_running_stats, process_group)

    def forward(self, x):
        """Do an inference on Identity."""
        return super().forward(x)


def conv_ws_2d(input,
               weight,
               bias=None,
               stride=1,
               padding=0,
               dilation=1,
               groups=1,
               eps=1e-5):
    """Conv2d with weight standarlization.

    :param input: input feature map
    :type input: torch.Tensor
    :param weight: weight of conv layer
    :type weight: torch.Tensor
    :param bias: bias
    :type bias: torch.Tensor
    :param stride: conv stride
    :type stride: int
    :param padding: num of padding
    :type padding: int
    :param dilation: num of dilation
    :type dilation: int
    :param groups: num of group
    :type groups: int
    :param eps: weight eps
    :type eps: float
    :return: feature map after weight standarlization
    :rtype: torch.Tensor
    """
    c_in = weight.size(0)
    weight_flat = weight.view(c_in, -1)
    mean_weight = weight_flat.mean(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    std = weight_flat.std(dim=1, keepdim=True).view(c_in, 1, 1, 1)
    weight = (weight - mean_weight) / (std + eps)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


@ClassFactory.register(ClassType.NETWORK)
class ConvWS2d(nn.Conv2d):
    """Conv2d with weight standarlization."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 eps=1e-5):
        """Init conv2d with weight standarlization.

        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: num of padding
        :param dilation: num of dilation
        :param groups: num of groups
        :param bias: bias
        :param eps: eps
        """
        super(ConvWS2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.eps = eps

    def forward(self, x):
        """Forward function of conv2d with weight standarlization."""
        return conv_ws_2d(x, self.weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups, self.eps)


@ClassFactory.register(ClassType.NETWORK)
class Flatten(OperatorSerializable, nn.Module):
    """Flatten Module."""

    def __init__(self, start_dim=0):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def forward(self, x):
        """Apply flatten."""
        old_shape = x.shape
        flatten_dim = old_shape[self.start_dim:]
        flatten_size = 1
        for i in range(len(flatten_dim)):
            flatten_size = flatten_size * flatten_dim[i]
        new_shape = old_shape[0:self.start_dim] + (flatten_size,)
        return torch.reshape(x, new_shape)


def expand(x, expand_shape):
    """Apply expand function."""
    expand_shape_new = []
    for size in expand_shape:
        if size == 1:
            size == -1
        expand_shape_new.append(size)
    return x.expand(expand_shape_new)


class Parameter(Torch_Parameter):
    """Wrapper of torch Parameter."""

    def __new__(cls, data=None, requires_grad=True, name=None):
        """Wrap __new__ of torch Parameter."""
        return Torch_Parameter.__new__(cls, data, requires_grad)


MSELoss = nn.MSELoss
Tensor = torch.Tensor
LayerNorm = torch.nn.LayerNorm
