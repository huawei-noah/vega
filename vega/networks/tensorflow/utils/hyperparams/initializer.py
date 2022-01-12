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

"""Defined initializer for tf backend."""
import tensorflow as tf
import tf_slim as slim
from object_detection.protos import hyperparams_pb2
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class Initializer(object):
    """Initializer."""

    def __init__(self, desc):
        """Init Initializer.

        :param desc: config dict
        """
        self.model = None
        self.type = desc.type if 'type' in desc else None
        self.mean = desc.mean if 'mean' in desc else 0.0
        self.stddev = desc.stddev if 'stddev' in desc else 0.01
        self.factor = 1.0
        self.uniform = True
        self.mode = desc.mode

    def get_real_model(self):
        """Get real model of initializer."""
        if self.model:
            return self.model
        else:
            if self.type == 'truncated_normal_initializer':
                self.model = tf.truncated_normal_initializer(
                    mean=self.mean, stddev=self.stddev)
            elif self.type == 'random_normal_initializer':
                self.model = tf.random_normal_initializer(
                    mean=self.mean, stddev=self.stddev)
            elif self.type == 'variance_scaling_initializer':
                enum_descriptor = (hyperparams_pb2.VarianceScalingInitializer.
                                   DESCRIPTOR.enum_types_by_name['Mode'])
                mode = self.mode
                if mode == 'FAN_IN':
                    mode = 0
                elif mode == 'FAN_OUT':
                    mode = 1
                elif mode == 'FAN_AVG':
                    mode = 2
                mode = enum_descriptor.values_by_number[mode].name
                self.mode = slim.variance_scaling_initializer(
                    factor=self.factor,
                    mode=mode,
                    uniform=self.uniform)
            else:
                self.model = None
                raise ValueError(
                    'Unknown initializer type: {}'.format(self.type))

            return self.model
