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

"""Defined faster rcnn detector."""
import tensorflow as tf
import tf_slim as slim

from object_detection.utils import context_manager
from object_detection.protos import hyperparams_pb2


def get_regularizer(desc):
    """Get regularizer function."""
    if desc.type == 'l1_regularizer':
        return slim.l1_regularizer(scale=float(desc.weight))
    elif desc.type == 'l2_regularizer':
        return slim.l2_regularizer(scale=float(desc.weight))
    else:
        raise ValueError('Unknown regularizer type: {}'.format(desc.type))


def get_initializer(desc):
    """Get initializer function."""
    mean = desc.mean if 'mean' in desc else 0.0
    stddev = desc.stddev if 'stddev' in desc else 0.01
    if desc.type == 'truncated_normal_initializer':
        return tf.truncated_normal_initializer(mean=mean, stddev=stddev)
    elif desc.type == 'random_normal_initializer':
        return tf.random_normal_initializer(mean=mean, stddev=stddev)
    elif desc.type == 'variance_scaling_initializer':
        enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.
                           DESCRIPTOR.enum_types_by_name['Mode'])
        mode = desc.mode
        if mode == 'FAN_IN':
            mode = 0
        elif mode == 'FAN_OUT':
            mode = 1
        elif mode == 'FAN_AVG':
            mode = 2

        mode = enum_descriptor.values_by_number[mode].name
        return slim.variance_scaling_initializer(
            factor=desc.factor,
            mode=mode,
            uniform=desc.uniform)
    else:
        raise ValueError('Unknown initializer type: {}'.format(desc.type))


def get_hyper_params_scope(desc):
    """Get hyper params scope."""
    op = desc.op
    affected_ops = [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose]
    if op and (op == hyperparams_pb2.Hyperparams.FC):
        affected_ops = [slim.fully_connected]

    def scope_fn():
        with context_manager.IdentityContextManager():
            with slim.arg_scope(
                    affected_ops,
                    weights_regularizer=get_regularizer(desc.regularizer),
                    weights_initializer=get_initializer(desc.initializer),
                    activation_fn=None,
                    normalizer_fn=None) as sc:
                return sc
    return scope_fn
