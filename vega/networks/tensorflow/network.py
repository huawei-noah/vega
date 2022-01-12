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
