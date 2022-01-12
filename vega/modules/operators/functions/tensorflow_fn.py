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

"""Custom functions of tensorflow."""
import logging
import math
from collections import OrderedDict
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.ops import state_ops
from vega.common.config import Config
from vega.common.class_factory import ClassType, ClassFactory
from vega.modules.operators.functions.serializable import OperatorSerializable
from vega.common.general import General


class Module(object):
    """Base Module to adapter tf Module."""

    def __init__(self):
        self.name = ''
        self.data_format = General.data_format
        self._modules = Config()
        self._parameters = OrderedDict()
        self._weights_buffer = OrderedDict()
        self._init_configs()

    def _init_configs(self):
        self._training = True
        self._trainable = True
        self.weight_file = None
        self.from_weight_type = None
        self._is_load_pretrained = False
        self.exclude_weight_prefix = None
        self._pre_hooks = OrderedDict()
        self._call_hooks = OrderedDict()

    def add_module(self, name, model):
        """Add models into self._models."""
        setattr(self, str(name), model)

    def build(self):
        """Build model or params."""
        pass

    def register_forward_pre_hook(self, hook):
        """Register pre hook."""
        self._pre_hooks[hook.__name__] = hook

    def register_forward_hook(self, hook):
        """Register call hook."""
        self._call_hooks[hook.__name__] = hook

    def named_modules(self):
        """Return names spaces."""
        self._apply_names()
        _modules = []
        for module in self.children():
            _modules.append((module.name, module))
            _modules.extend(module.named_modules())
        return _modules

    def named_children(self):
        """Return names children."""
        return [(name, module) for name, module in self._modules.items()]

    def children(self):
        """Get child models of current Module."""
        for model in self._modules.values():
            yield model

    def load_checkpoint(self, weight_file):
        """Load weight state dict from last checkpoint file."""
        if not weight_file:
            return
        logging.info("Load checkpoint form file ({}).".format(weight_file))
        reader = tf.train.NewCheckpointReader(weight_file)
        variables = reader.get_variable_to_shape_map()
        states = {v: reader.get_tensor(v) for v in variables}
        self.load_checkpoint_from_numpy(states)

    def load_checkpoint_from_numpy(self, states):
        """Load checkpoint from numpy."""
        states = self._exclude_checkpoint_by_prefix(states)
        for name, module in self.named_modules():
            child_state = [(k, v) for k, v in states.items() if k.startswith(module.name + '/')]
            for k, v in child_state:
                module.set_weights(k, v)

    def _exclude_checkpoint_by_prefix(self, states):
        if self.exclude_weight_prefix:
            if not isinstance(self.exclude_weight_prefix, list):
                self.exclude_weight_prefix = [self.exclude_weight_prefix]
            for prefix in self.exclude_weight_prefix:
                states = {k: v for k, v in states.items() if not k.startswith(prefix)}
        return states

    def set_weights(self, name, value):
        """Set weights into weights buffer."""
        self._weights_buffer[name] = value

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

    def freeze(self):
        """Set training flag."""
        self._trainable = False
        for module in self.children():
            module.freeze()

    def __setattr__(self, key, value):
        """Set name to modules."""
        super().__setattr__(key, value)
        if isinstance(value, Module):
            self._modules[key] = value

    def set_parameters(self, name, value):
        """Set Parameters."""
        self._parameters[name] = value
        setattr(self, name, value)
        return self.name

    def get_weights(self, name=None):
        """Get weights by name."""
        if name is None:
            return self._weights_buffer
        else:
            return tf.get_default_graph().get_tensor_by_name('{}:0'.format(name))

    def get_all_weights(self):
        """Get all weights."""
        all_weights = OrderedDict()
        for child in self.children():
            all_weights.update(child._weights_buffer)
            if isinstance(child, Module):
                all_weights.update(child.get_all_weights())
        return all_weights

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

    def _apply_names(self, parent_name=''):
        """Apply names spaces."""
        for scope_name, module in self._modules.items():
            scope_name = '{}.{}'.format(parent_name, scope_name) if parent_name else scope_name
            module.name = module.name or scope_name + '/' + module.__class__.__name__
            module._apply_names(scope_name)

    def _apply_parameters(self):
        """Apply names spaces."""
        for name, params in self._parameters.items():
            setattr(self, name, tf.Variable(params, name='{}.{}'.format(self.name, name) if self.name else name))

    def __call__(self, inputs, *args, **kwargs):
        """Call call function."""
        self.build()
        self._apply_parameters()
        self._apply_names()
        for module in self.children():
            module._is_load_pretrained = True
        if self.training:
            for hook_name, hook in self._pre_hooks.items():
                inputs = hook(self, inputs) or inputs
        out = self.call(inputs, *args, **kwargs)
        if self.training:
            for hook_name, hook in self._call_hooks.items():
                out = hook(self, inputs, out) or out
        self._apply_weights()
        return out

    def _apply_weights(self):
        if not self._weights_buffer:
            return
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)
        if isinstance(self, Conv2d):
            self._weights_buffer = {k.replace('/weights', '/kernel'): v for k, v in self._weights_buffer.items()}
        values = [(var, self._weights_buffer.get(var.name.replace(':0', ''))) for var in variables if
                  var.name.replace(':0', '') in self._weights_buffer]
        for v, weight in values:
            if len(v.shape) == 4:
                if v.shape[2] != weight.shape[2]:
                    import torch
                    num = v.shape[2] // weight.shape[2]
                    weight = torch.cat([weight] * num, 2)
            v._initializer_op = state_ops.assign(v, weight)
        self._weights_buffer.clear()

    def modules(self):
        """Get the current modules."""
        if self._modules.values():
            return self._modules.values()
        else:
            return [self]


@ClassFactory.register(ClassType.NETWORK)
class QuantizeConv2d(OperatorSerializable):
    """QuantizeConv2d Module inherit nn.Module."""

    def __init__(self):
        """Construct Identity class."""
        OperatorSerializable.__init__(self)

    def call(self, inputs, **kwargs):
        """Call QuantizeConv2d function."""
        return inputs


@ClassFactory.register(ClassType.NETWORK)
class Pad(Module, OperatorSerializable):
    """Pad layer."""

    def __init__(self, kernel_size):
        super(Pad, self).__init__()
        self.kernel_size = kernel_size

    def call(self, inputs, *args, **kwargs):
        """Call padding function."""
        return inputs


class HeInitial(object):
    """Initialize of Hekaiming."""

    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, tensor, **kwargs):
        """Call He_initial function."""
        c, h, w = get_shape(tensor)[1:]
        fan_in = c * h * w
        std = math.sqrt(2) / math.sqrt(fan_in)
        return tf.random_normal_initializer(0, std * self.scale)


@ClassFactory.register(ClassType.NETWORK)
class Conv2d(Module, OperatorSerializable):
    """Fuse and unified conv2d args."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True, groups=1,
                 dilation=1, separable=False, depthwise=False, padding_mode='same', bn=False):
        super(Conv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.groups = groups
        self.dilation = dilation
        self.kernel_initial = tf.variance_scaling_initializer()
        self.bias_initial = tf.zeros_initializer()
        self._initializer = None
        self.reuse = None
        self.separable = separable
        self.depthwise = depthwise
        self.padding_mode = padding if isinstance(padding, str) and stride != 2 else padding_mode
        self.bn = bn

    def call(self, inputs, **kwargs):
        """Call separable_conv2d function."""
        if self._initializer:
            self.kernel_initial = self._initializer(inputs)
        if self.dilation > 1:
            conv2d = tf.keras.layers.SeparableConv2D(filters=self.out_channels,
                                                     kernel_size=self.kernel_size,
                                                     strides=self.stride,
                                                     data_format=self.data_format,
                                                     dilation_rate=self.dilation,
                                                     padding=self.padding_mode,
                                                     use_bias=self.bias,
                                                     name=self.name, trainable=self._trainable)
        else:
            conv2d = tf.keras.layers.Conv2D(filters=self.out_channels,
                                            kernel_size=self.kernel_size,
                                            kernel_initializer=self.kernel_initial,
                                            bias_initializer=self.bias_initial,
                                            strides=self.stride,
                                            data_format=self.data_format,
                                            dilation_rate=self.dilation,
                                            padding=self.padding_mode,
                                            use_bias=self.bias,
                                            name=self.name, trainable=self._trainable)
        x = conv2d(inputs=inputs)
        if self.bn:
            bn = BatchNorm2d(name=self.name + '/BatchNorm')
            x = bn(x)
        return x

    def initial(self, kernel_mode='he', bias_mode='zero', kernel_scale=1., bias_scale=1.):
        """Initialize weight and bias."""
        if kernel_mode == 'he':
            self._initializer = HeInitial(kernel_scale)


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

    def call(self, input, **kwargs):
        """Call separable_conv2d function."""
        model = tf.keras.layers.SeparableConv2D(filters=self.out_channels,
                                                kernel_size=self.kernel_size,
                                                strides=self.stride,
                                                data_format=self.data_format,
                                                dilation_rate=self.dilation,
                                                depthwise_initializer=tf.variance_scaling_initializer(),
                                                pointwise_initializer=tf.variance_scaling_initializer(),
                                                padding='SAME', use_bias=self.bias,
                                                name=self.name,
                                                reuse=self.reuse, trainable=self._trainable)

        return model(inputs=input)


@ClassFactory.register(ClassType.NETWORK)
class MaxPool2d(Module, OperatorSerializable):
    """Fuse and unified MaxPool2d args."""

    def __init__(self, kernel_size, stride, padding='SAME'):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def call(self, input, **kwargs):
        """Call MaxPooling2D function."""
        model = tf.layers.MaxPooling2D(pool_size=self.kernel_size, strides=self.stride,
                                       data_format=self.data_format, padding=self.padding, name=self.name,
                                       trainable=self._trainable)
        x = model(inputs=input)
        return x


@ClassFactory.register(ClassType.NETWORK)
class Zero(Module, OperatorSerializable):
    """Class of Zero operation."""

    def __init__(self, stride):
        """Init Zero."""
        super(Zero, self).__init__()
        self.stride = stride

    def call(self, x, **kwargs):
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

    def call(self, inputs, **kwargs):
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

    def call(self, input, **kwargs):
        """Call relu function."""
        return tf.nn.relu(input)


@ClassFactory.register(ClassType.NETWORK)
class Relu6(Module, OperatorSerializable):
    """Call relu6."""

    def __init__(self, inplace=False):
        super(Relu6, self).__init__()
        self.inplace = inplace

    def call(self, input, **kwargs):
        """Call relu6 function."""
        return tf.nn.relu6(input)


@ClassFactory.register(ClassType.NETWORK)
class Hswish(Module, OperatorSerializable):
    """Call Hswish."""

    def __init__(self, inplace=False):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def call(self, input, **kwargs):
        """Call Hswish function."""
        return input * tf.nn.relu6(input + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class Hsigmoid(Module, OperatorSerializable):
    """Call Hsigmoid."""

    def __init__(self, inplace=False):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def call(self, input, **kwargs):
        """Call Hsigmoid function."""
        return tf.nn.relu6(input + 3.) / 6.


@ClassFactory.register(ClassType.NETWORK)
class AdaptiveAvgPool2d(Module, OperatorSerializable):
    """Call reduce_mean."""

    def __init__(self, output_size=(1, 1)):
        super(AdaptiveAvgPool2d, self).__init__()
        self.output_size = output_size

    def call(self, input, **kwargs):
        """Call reduce_mean function."""
        axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
        return tf.reduce_mean(input, axes, keepdims=True)


@ClassFactory.register(ClassType.NETWORK)
class Linear(Module, OperatorSerializable):
    """Call dense."""

    def __init__(self, in_features=None, out_features=None, use_bias=True, activation=None):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

    def call(self, input, **kwargs):
        """Call dense function."""
        if len(input.shape) == 4:
            input = tf.squeeze(input, axis=[2, 3])
        fc = tf.keras.layers.Dense(units=self.out_features, use_bias=self.use_bias, name=self.name,
                                   activation=self.activation)
        out = fc(inputs=input)
        return out


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

    def call(self, input, **kwargs):
        """Call average_pooling2d function."""
        return tf.keras.layers.AveragePooling2D(pool_size=self.kernel_size,
                                                strides=self.stride,
                                                data_format=self.data_format,
                                                padding='SAME',
                                                name=self.name, trainable=self._trainable)(input)


@ClassFactory.register(ClassType.NETWORK)
class BatchNorm2d(Module, OperatorSerializable):
    """Call batch_normalization."""

    def __init__(self, num_features=None, eps=1e-05, momentum=0.997, affine=None, name=None):
        super(BatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.training = affine if affine is not None else self.training
        self.affine = affine
        self.name = name

    def call(self, input, **kwargs):
        """Call batch_normalization function."""
        bn = tf.keras.layers.BatchNormalization(momentum=self.momentum,
                                                axis=1 if self.data_format == 'channels_first' else 3,
                                                epsilon=self.eps,
                                                center=True, scale=True, fused=True,
                                                name=self.name, trainable=self._trainable)
        if self._is_load_pretrained:
            self.training = True
        out = bn(inputs=input, training=self.training)
        if self._trainable:
            for item in bn.updates:
                tf.add_to_collections(tf.GraphKeys.UPDATE_OPS, item)
        return out


@ClassFactory.register(ClassType.NETWORK)
class Identity(Module, OperatorSerializable):
    """Class of Identity operation."""

    def __init__(self):
        """Init Identity."""
        super(Identity, self).__init__()

    def call(self, x, **kwargs):
        """Forward function of Identity."""
        return tf.identity(x)


@ClassFactory.register(ClassType.NETWORK)
class Dropout(Module, OperatorSerializable):
    """Class of Dropout."""

    def __init__(self, prob=0.5, inplace=False):
        """Construct Dropout class."""
        super(Dropout, self).__init__()
        self.dropout = tf.keras.layers.Dropout(prob)

    def call(self, x, **kwargs):
        """Call Dropout function."""
        out = self.dropout(x)
        return out


@ClassFactory.register(ClassType.NETWORK)
class Tanh(Module, OperatorSerializable):
    """Class of Dropout."""

    def call(self, x, **kwargs):
        """Forward Tanh."""
        return super(Tanh, self).forward(x)


@ClassFactory.register(ClassType.NETWORK)
class Embedding(Module, OperatorSerializable):
    """Class of Embedding."""

    def __init__(self, num_embeddings, embedding_dim):
        super(Embedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(num_embeddings, embedding_dim, )

    def call(self, x, **kwargs):
        """Call embedding."""
        return self.embedding(x)


@ClassFactory.register(ClassType.NETWORK)
class PixelShuffle(Module, OperatorSerializable):
    """Class of PixelShuffle."""

    def __init__(self, upscale):
        super(PixelShuffle, self).__init__()
        self.upscale = upscale

    def call(self, inputs, **kwargs):
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

    def call(self, inputs, **kwargs):
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

    def call(self, inputs, **kwargs):
        """Forward function of squeeze."""
        return tf.squeeze(inputs, [self.dim])


@ClassFactory.register(ClassType.NETWORK)
class Permute(Module, OperatorSerializable):
    """Class of Permute."""

    def __init__(self, size=None):
        super(Permute, self).__init__()
        self.size = size

    def call(self, inputs, **kwargs):
        """Forward function of Permute."""
        return tf.transpose(inputs, self.size)


@ClassFactory.register(ClassType.NETWORK)
class Stack(Module, OperatorSerializable):
    """Class of Stack."""

    def __init__(self, dim=0):
        super(Stack, self).__init__()
        self.dim = dim

    def call(self, inputs, **kwargs):
        """Forward function of Stack."""
        return tf.stack(inputs, self.dim)


@ClassFactory.register(ClassType.NETWORK)
class Transpose(Module, OperatorSerializable):
    """Class of Transpose."""

    def __init__(self, dim1=0, dim2=1):
        super(Transpose, self).__init__()
        self.dim1, self.dim2 = dim1, dim2

    def call(self, inputs, **kwargs):
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

    def call(self, input, **kwargs):
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

    def call(self, inputs, **kwargs):
        """Call InterpolateScale."""
        inputs = tf.transpose(inputs, [0, 2, 3, 1])
        if self.size is not None:
            if isinstance(self.size, int):
                self.size = (self.size, self.size)
            output = tf.image.resize(inputs, size=self.size, method=self.mode, align_corners=self.align_corners)
        else:
            if self.scale_factor is not None:
                output = tf.image.resize_images(inputs, [inputs.shape[1] * self.scale_factor,
                                                         inputs.shape[2] * self.scale_factor], method=self.mode,
                                                align_corners=self.align_corners)
            else:
                output = tf.image.resize_images(inputs, (tf.cast(inputs.shape[1], tf.int32),
                                                         tf.cast(inputs.shape[2], tf.int32)), method=self.mode,
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

    def call(self, inputs, *args, **kwargs):
        """Call MeanShift."""
        std = tf.convert_to_tensor(self.rgb_std, dtype=tf.float32)
        self.weight = tf.convert_to_tensor(np.eye(3).astype(np.float32))
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


def matmul(a, b):
    """Call matmul according to backends."""
    return tf.matmul(a, b)


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
    len_dim = len(list(inputs.shape))
    pos = [[0, 0] for i in range(len_dim)]
    if isinstance(inputs, tf.Tensor):
        for i in range(len(position)):
            if i % 2 == 0:
                pos[(-(i // 2) - 1)][0] = position[i]
            else:
                pos[(-(i // 2) - 1)][1] = position[i]
            return tf.pad(inputs, pos)
    else:
        for i in range(len(position)):
            if i % 2 == 0:
                pos[((i // 2))][0] = position[i]
            else:
                pos[((i // 2))][1] = position[i]
    for i, v in enumerate(pos):
        if v[1] < 0:
            inputs = np.delete(inputs, list(range(-v[1])), axis=i)
            pos[i] = (0, 0)
    return np.pad(inputs, pos)


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


def ones(input_size):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return tf.ones(input_size)


def one_hot(inputs, num_classes):
    """Take LongTensor with index values of shape."""
    return tf.one_hot(inputs, num_classes)


def ones_like(out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return tf.ones_like(out)


def zeros_like(out):
    """Return a tensor with all 1s. The shape is defined by the variable parameter size."""
    return tf.zeros_like(out)


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


def gelu(x):
    """Apply gelu function."""
    return x * 0.5 * (1.0 + tf.erf(x / math.sqrt(2.0)))


def swish(x):
    """Apply swish function."""
    return x * tf.sigmoid(x)


def relu(x):
    """Apply relu function."""
    return tf.nn.relu(x)


def sqrt(x):
    """Apply sqrt function."""
    return tf.sqrt(x)


@ClassFactory.register(ClassType.NETWORK)
class LayerNorm(Module, OperatorSerializable):
    """Layer Norm module."""

    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = self.set_parameters('gamma', ones(hidden_size))
        self.bias = self.set_parameters('beta', zeros(hidden_size))
        self.variance_epsilon = eps

    def call(self, x):
        """Call LayerNorm."""
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


@ClassFactory.register(ClassType.NETWORK)
class Flatten(Module, OperatorSerializable):
    """Flatten Module."""

    def __init__(self, start_dim=0):
        super(Flatten, self).__init__()
        self.start_dim = start_dim

    def call(self, x):
        """Apply flatten."""
        old_shape = x.shape
        flatten_dim = old_shape[self.start_dim:]
        flatten_size = 1
        for i in range(len(flatten_dim)):
            flatten_size = flatten_size * flatten_dim[i]
        new_shape = old_shape[0:self.start_dim] + (flatten_size,)
        return tf.reshape(x, new_shape)


class Tensor():
    """Wrapper of Tensor."""

    pass


class Parameter():
    """Wrapper of Parameter."""

    pass


def expand(x, expand_shape):
    """Expand function."""
    pass


def MSELoss():
    """MSE Loss."""
    pass
