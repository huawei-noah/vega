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
from official.r1.resnet import resnet_model
from vega.common import ClassType, ClassFactory
import tf_slim.nets.resnet_v1 as resnet_v1
import tf_slim.nets.resnet_v2 as resnet_v2
import tensorflow as tf


@ClassFactory.register(ClassType.NETWORK)
class ResNetTF(resnet_model.Model):
    """Model class with appropriate defaults for Imagenet data."""

    def __init__(self, depth=50, data_format='channels_first', num_classes=1001, resnet_version=1,
                 dtype=resnet_model.DEFAULT_DTYPE):
        if depth < 50:
            bottleneck = False
        else:
            bottleneck = True
        super(ResNetTF, self).__init__(
            resnet_size=depth,
            bottleneck=bottleneck,
            num_classes=num_classes,
            num_filters=64,
            kernel_size=7,
            conv_stride=2,
            first_pool_size=3,
            first_pool_stride=2,
            block_sizes=_get_block_sizes(depth),
            block_strides=[1, 2, 2, 2],
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )


def _get_block_sizes(resnet_size):
    """Retrieve the size of each block_layer in the ResNet model."""
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    return choices[resnet_size]


@ClassFactory.register(ClassType.NETWORK)
class ResNetSlim(object):
    """ResNetSlim class."""

    def __init__(self, depth=50, num_classes=1000, version='v1'):
        super(ResNetSlim, self).__init__()
        self.depth = depth
        self.version = version
        self.num_classes = num_classes

    def __call__(self, inputs, training=True):
        """Call ResNet."""
        resnet = create_resnet_from_slim(self.depth, self.version)
        out = resnet(inputs, self.num_classes, training)
        return tf.squeeze(out[0], axis=[1, 2])


def create_resnet_from_slim(depth, version):
    """Create from slim."""
    pkg = resnet_v1 if version == 'v1' else resnet_v2
    cls_name = 'resnet_{}_{}'.format(version, depth)
    return getattr(pkg, cls_name)
