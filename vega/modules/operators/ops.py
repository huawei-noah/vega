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

"""unified operators."""
from functools import partial
from vega.common.class_factory import ClassFactory, ClassType
from vega import is_tf_backend, is_ms_backend

if is_tf_backend():
    from .functions import tensorflow_fn as fn
elif is_ms_backend():
    from .functions import mindspore_fn as fn
else:
    from .functions import pytorch_fn as fn

    ConvWS2d = fn.ConvWS2d
    GroupNorm = fn.GroupNorm
    SyncBatchNorm = fn.SyncBatchNorm
    ConvTranspose2d = fn.ConvTranspose2d

Module = fn.Module
Conv2d = fn.Conv2d
QuantizeConv2d = fn.QuantizeConv2d
SeparableConv2d = fn.SeparableConv2d
BatchNorm2d = fn.BatchNorm2d
MaxPool2d = fn.MaxPool2d
AvgPool2d = fn.AvgPool2d
AdaptiveAvgPool2d = fn.AdaptiveAvgPool2d
Identity = fn.Identity
Zero = fn.Zero
create_zeros = fn.zeros
zeros = fn.zeros
Relu = fn.Relu
Relu6 = fn.Relu6
Hswish = fn.Hswish
Hsigmoid = fn.Hsigmoid
Linear = fn.Linear
Pad = fn.Pad
View = fn.View
concat = fn.concat
mul = fn.mul
random_normal = fn.random_normal
softmax = fn.softmax
to_numpy = fn.to_numpy
mean = fn.mean
tensor_abs = fn.tensor_abs
mean_all = fn.mean_all
pad = fn.pad
interpolate = fn.interpolate
add_n = fn.add_n
get_shape = fn.get_shape
drop_path = fn.drop_path
MoudleList = fn.MoudleList
PixelShuffle = fn.PixelShuffle
Split = fn.Split
Squeeze = fn.Squeeze
Permute = fn.Permute
Stack = fn.Stack
Transpose = fn.Transpose
InterpolateScale = fn.InterpolateScale
LeakyReLU = fn.LeakyReLU
MeanShift = fn.MeanShift
GlobalMaxPool1d = fn.GlobalMaxPool1d
maximum = fn.maximum
minimum = fn.minimum
new_constant = fn.new_constant
argmax = fn.argmax
clamp = fn.clamp
where = fn.where
unique = fn.unique
log = fn.log
convert_to_tensor = fn.convert_to_tensor
new_ones = fn.new_ones
arange = fn.arange
compare_where = fn.compare_where
unsqueeze = fn.unsqueeze
expand_as = fn.expand_as
exp = fn.exp
gumbel_softmax = fn.gumbel_softmax
pow = fn.pow
ones = fn.ones
one_hot = fn.one_hot
reduce_sum = fn.reduce_sum
to = fn.to

Dropout = fn.Dropout
Tanh = fn.Tanh
matmul = fn.matmul
gelu = fn.gelu
swish = fn.swish
relu = fn.relu
Embedding = fn.Embedding
sqrt = fn.sqrt
ones_like = fn.ones_like
zeros_like = fn.zeros_like
LayerNorm = fn.LayerNorm
Tensor = fn.Tensor
Parameter = fn.Parameter
Flatten = fn.Flatten
expand = fn.expand
MSELoss = fn.MSELoss


def from_module(module):
    """From Model."""
    name = module.__class__.__name__
    if ClassFactory.is_exists(ClassType.NETWORK, name):
        module_cls = ClassFactory.get_cls(ClassType.NETWORK, name)
        if hasattr(module_cls, "from_module"):
            return module_cls.from_module(module)
    return module


@ClassFactory.register(ClassType.NETWORK)
class Lambda(Module):
    """Lambda Module.

    :Example:
         >>> def multiply(x, y): return x + y
         >>> Lambda(multiply)
    """

    def __init__(self, func, *args, **kwargs):
        super(Lambda, self).__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def call(self, inputs):
        """Override call function."""
        return partial(self.func, *self.args, **self.kwargs)(inputs)
