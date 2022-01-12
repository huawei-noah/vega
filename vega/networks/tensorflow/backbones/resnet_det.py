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

"""Residual Network."""
import tensorflow as tf
from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as frcnn_resnet_v1
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class ResNetDet(object):
    """ResNet basic Class."""

    def __init__(self, desc):
        """Init ResNetDet.

        :param desc: config dict
        """
        self.resnet_size = desc.resnet_size
        self.resnet_version = desc.resnet_version
        if self.resnet_version not in (1, 2):
            raise ValueError('Resnet version should be 1 or 2.')

        if self.resnet_size == 50:
            self.feature_extractor = frcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor
        elif self.resnet_size == 101:
            self.feature_extractor = frcnn_resnet_v1.FasterRCNNResnet101FeatureExtractor
        elif self.resnet_size == 152:
            self.feature_extractor = frcnn_resnet_v1.FasterRCNNResnet152FeatureExtractor
        else:
            raise ValueError('Resnet version should be 1 or 2.')

        self.first_stage_features_stride = desc['first_stage_features_stride']
        self.inplace_batchnorm_update = desc['inplace_batchnorm_update']
        self.batch_norm_trainable = desc['batch_norm_trainable']
        self.fp16 = False
        self.model = None

    def get_real_model(self, training):
        """Get real model of ResnetDet."""
        if self.model:
            return self.model
        else:
            self.model = self.feature_extractor(
                training, self.first_stage_features_stride,
                batch_norm_trainable=self.batch_norm_trainable)
            return self.model

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=tf.float32,
                             *args, **kwargs):
        """Create variables in fp32, then casts to fp16 if necessary."""
        if self.fp16 and dtype == tf.float16:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        """Return a variable scope of created model."""
        return tf.variable_scope('resnet_model',
                                 custom_getter=self._custom_dtype_getter)

    def __call__(self, features, labels, training):
        """Forward function of ResNetDet."""
        return self.get_real_model(training).predict(features, labels)
