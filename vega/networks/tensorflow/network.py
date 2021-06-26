# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined TensorFlow sequential network."""
import tensorflow as tf


class Sequential(object):
    """Sequential network for TensorFlow."""

    def __init__(self, modules):
        """Init Sequential."""
        self.modules = modules

    def __call__(self, inputs, training):
        """Forward function for sequential network."""
        for modules in self.modules:
            inputs = modules(inputs, training)
        return inputs


class Network(object):
    """Base class for tensorflow Network."""

    def __init__(self, desc):
        """Init network."""
        self.fp16 = tf.float16 if desc.get('fp16', False) else tf.float32

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float16,
                             *args, **kwargs):
        """Create variables in fp32, then casts to fp16 if necessary."""
        if self.fp16 and dtype == tf.float16:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self, scope_name):
        """Return a variable scope of created model."""
        return tf.variable_scope(scope_name, custom_getter=self._custom_dtype_getter)
