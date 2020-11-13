# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""unified operators."""
from functools import partial
from zeus.common.class_factory import ClassFactory, ClassType
from zeus import is_tf_backend, is_ms_backend

if is_tf_backend():
    from .functions import tensorflow_fn as fn
elif is_ms_backend():
    from .functions import mindspore_fn as fn
else:
    from .functions import pytorch_fn as fn

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
Relu = fn.Relu
Relu6 = fn.Relu6
Hswish = fn.Hswish
Hsigmoid = fn.Hsigmoid
Linear = fn.Linear
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
