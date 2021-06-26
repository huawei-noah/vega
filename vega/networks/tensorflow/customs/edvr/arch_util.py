# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""arch util modules."""
import math
import tensorflow as tf


def resize(x, size, align_corners=False, name=None, half_pixel_centers=False, method='bicubic'):
    """Resize function."""
    if method == 'bicubic':
        upsampling = tf.image.resize_bicubic
    elif method == 'bilinear':
        upsampling = tf.image.resize_bilinear
    else:
        raise ValueError
    return upsampling(x, size=size, align_corners=align_corners, name=name, half_pixel_centers=half_pixel_centers)


def calculate_gain(nonlinearity, param=None):
    """Calculate gain for linear functions."""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leakyrelu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def calculate_fan(kernel_size, in_channels, out_channels=None, mode='fan_in'):
    """Calculate fan value."""
    if mode == 'fan_in':
        fan = in_channels
    elif mode == 'fan_out':
        fan = out_channels
    else:
        raise KeyError
    for k in kernel_size:
        fan *= k
    return fan


def get_initializer(init_cfg, in_channels, out_channels, kernel_size):
    """Get initializer of random method."""
    type = init_cfg.pop('type')

    if type == 'kaiming_uniform':
        a = init_cfg.pop('a', 0)
        mode = init_cfg.pop('mode', 'fan_in')
        nonlinearity = init_cfg.pop('nonlinearity', 'leakyrelu')
        fan = calculate_fan(kernel_size, in_channels, out_channels, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        initializer = tf.random_uniform_initializer(-bound, bound)
    elif type == 'kaiming_normal':
        a = init_cfg.pop('a', 0)
        mode = init_cfg.pop('mode', 'fan_in')
        nonlinearity = init_cfg.pop('nonlinearity', 'leakyrelu')
        fan = calculate_fan(kernel_size, in_channels, out_channels, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        initializer = tf.random_normal_initializer(0.0, std)
    elif type == 'xavier_uniform':
        gain = init_cfg.pop('gain', 1.)
        fan_in = calculate_fan(kernel_size, in_channels, out_channels, 'fan_in')
        fan_out = calculate_fan(kernel_size, in_channels, out_channels, 'fan_out')
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        initializer = tf.random_uniform_initializer(-a, a)
    elif type == 'xavier_normal':
        gain = init_cfg.pop('gain', 1.)
        fan_in = calculate_fan(kernel_size, in_channels, out_channels, 'fan_in')
        fan_out = calculate_fan(kernel_size, in_channels, out_channels, 'fan_out')
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        initializer = tf.random_normal_initializer(0.0, std)
    else:
        raise NotImplementedError

    return initializer


def pair(x, dims=2):
    """List pair in dimensions."""
    if isinstance(x, list) or isinstance(x, tuple):
        if len(x) != dims:
            raise Exception('length of x must equal to dims.')
    elif isinstance(x, int):
        x = [x] * dims
    else:
        raise ValueError
    return x


def Conv2D(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=tf.zeros_initializer(), trainable=True, name='Conv2D'):
    """Convolution 2D layer."""
    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv2d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x


def ReLU(x, name=None):
    """Activation layer of ReLU."""
    x = tf.nn.relu(x, name=name)
    return x


def LeakyReLU(x, alpha=0.1, name=None):
    """Leaky ReLU activation layer."""
    x = tf.nn.leaky_relu(x, alpha=alpha, name=name)
    return x


class ActLayer(object):
    """Activation layer."""

    def __init__(self, cfg, name=None):
        super(ActLayer, self).__init__()
        self.type = cfg.get('type').lower()
        if self.type == 'leakyrelu':
            self.alpha = cfg.get('alpha', 0.2)
        self.name = name

    def _forward(self, x):
        if self.type == 'relu':
            return ReLU(x, name=self.name)
        elif self.type == 'leakyrelu':
            return LeakyReLU(x, alpha=self.alpha, name=self.name)
        else:
            raise NotImplementedError

    def __call__(self, x):
        """Forward function of act layer."""
        shape = list(map(int, x.shape))
        if len(shape) == 5:
            # TODO
            # Ascend currently do not support 5D relu
            x_4d = tf.reshape(x, [-1] + shape[2:])
            x_4d = self._forward(x_4d)
            x = tf.reshape(x_4d, shape)
        else:
            x = self._forward(x)

        return x


def ConvModule(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', dilations=(1, 1), use_bias=True,
               kernel_initializer=None, bias_initializer=None, act_cfg=None, trainable=True, name='Conv2D'):
    """Convolution and activation module."""
    if act_cfg is not None:
        nonlinearity = act_cfg.get('type').lower()
        if nonlinearity == 'leakyrelu':
            a = act_cfg.get('alpha', 0.01)
        else:
            nonlinearity = 'relu'
            a = 0
        if kernel_initializer is None:
            kernel_initializer = get_initializer(
                dict(type='kaiming_uniform', a=a, nonlinearity=nonlinearity), int(x.shape[-1]), filters, kernel_size)

    x = Conv2D(x, filters, kernel_size, strides, padding, dilations, use_bias,
               kernel_initializer=kernel_initializer, bias_initializer=None,
               trainable=True, name=name)

    if act_cfg is not None:
        x = ActLayer(act_cfg)(x)

    return x


def depth_to_space(x, scale, use_default=False):
    """Depth to space function."""
    if use_default:
        out = tf.depth_to_space(x, scale)
    else:
        b, h, w, c = list(map(int, x.shape))
        out = tf.reshape(x, [b, h, w, scale, scale, -1])
        out = tf.transpose(out, [0, 1, 3, 2, 4, 5])
        out = tf.reshape(out, [b, h * scale, w * scale, -1])
    return out


def tf_split(x, num_or_size_splits, axis=0, num=None, keep_dims=False):
    """Split feature map of high dimension into list of feature map of low dimension."""
    x_list = tf.split(x, num_or_size_splits, axis, num)

    if not keep_dims:
        x_list2 = [tf.squeeze(x_, axis) for x_ in x_list]
        return x_list2

    return x_list


class ResBlockNoBN(object):
    """Residual block with batch norm."""

    def __init__(self, num_blocks, mid_channels, res_scale=1.0,
                 act_cfg=dict(type='ReLU'), trainable=True, name='ResBlock'):
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.res_scale = res_scale
        self.name = name
        self.trainable = trainable
        self.act_cfg = act_cfg

    def build_block(self, x, idx):
        """Build the residual block without bn."""
        fan_in = int(x.shape[-1])
        out = Conv2D(x, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1 / (100 * fan_in))),
                     trainable=self.trainable, name='conv{}a'.format(idx))
        out = ActLayer(self.act_cfg)(out)
        out = Conv2D(out, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1 / (100 * fan_in))),
                     trainable=self.trainable, name='conv{}b'.format(idx))
        return x + out * self.res_scale

    def __call__(self, x):
        """Forward function of block."""
        with tf.variable_scope(self.name):
            for i in range(self.num_blocks):
                x = self.build_block(x, i + 1)
            return x


def Conv3D(x, filters, kernel_size=(1, 1, 1), strides=(1, 1, 1), padding='same', dilations=(1, 1, 1), use_bias=True,
           kernel_initializer=None, bias_initializer=tf.zeros_initializer(),
           trainable=True, name='Conv3D'):
    """Convolution 3D layer."""
    if kernel_initializer is None:
        kernel_initializer = get_initializer(
            dict(type='kaiming_uniform', a=math.sqrt(5)), int(x.shape[-1]), filters, kernel_size)
    if bias_initializer is None:
        fan = calculate_fan(kernel_size, int(x.shape[-1]))
        bound = 1 / math.sqrt(fan)
        bias_initializer = tf.random_uniform_initializer(-bound, bound)

    x = tf.layers.conv3d(
        x,
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding.lower(),
        dilation_rate=dilations,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        trainable=trainable,
        name=name,
    )
    return x


class ResBlockChnAtten(object):
    """Residual block with channels attention."""

    def __init__(self, num_blocks, mid_channels, res_scale=1.0,
                 act_cfg=dict(type='ReLU'), trainable=True, name='ResBlock'):
        self.num_blocks = num_blocks
        self.mid_channels = mid_channels
        self.res_scale = res_scale
        self.name = name
        self.trainable = trainable
        self.act_cfg = act_cfg

    def build_block(self, x, idx):
        """Build the block."""
        fan_in = int(x.shape[-1])
        out = Conv2D(x, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1 / (100 * fan_in))),
                     trainable=self.trainable, name='conv{}a'.format(idx))
        out = ActLayer(self.act_cfg)(out)
        out = Conv2D(out, self.mid_channels,
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1 / (100 * fan_in))),
                     trainable=self.trainable, name='conv{}b'.format(idx))
        B, H, W, C = out.get_shape().as_list()
        chn_atten = tf.nn.avg_pool2d(out, ksize=[H, W], strides=1, padding='VALID')
        chn_atten = tf.reshape(chn_atten, [-1, C])
        chn_atten = tf.layers.dense(chn_atten, C)
        chn_atten = ActLayer(self.act_cfg)(chn_atten)
        chn_atten = tf.sigmoid(tf.layers.dense(chn_atten, C))
        chn_atten = tf.math.multiply(out, tf.reshape(chn_atten, [B, 1, 1, C]))
        out = tf.concat([out, chn_atten], axis=-1)
        out = Conv2D(out, self.mid_channels, kernel_size=(1, 1),
                     kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=math.sqrt(1 / (100 * fan_in))),
                     trainable=self.trainable, name='conv{}c'.format(idx))
        return out + x

    def __call__(self, x):
        """Forward function of residual block with channels attention."""
        with tf.variable_scope(self.name):
            for i in range(self.num_blocks):
                x = self.build_block(x, i + 1)
            return x
