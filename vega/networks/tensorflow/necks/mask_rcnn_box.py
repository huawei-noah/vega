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

from object_detection.predictors.heads import box_head
from object_detection.predictors import mask_rcnn_box_predictor
from object_detection.predictors.heads import class_head
from vega.common import ClassType, ClassFactory
from vega.networks.tensorflow.utils.hyperparams import scope_generator


@ClassFactory.register(ClassType.NETWORK)
class MaskRCNNBox(object):
    """Mask RCNN Box."""

    def __init__(self, desc):
        """Init MaskRCNNBox.

        :param desc: config dict
        """
        self.model = None
        self.num_classes = desc.num_classes
        self.add_background_class = desc.add_background_class if 'add_background_class' in desc else True
        self.num_class_slots = self.num_classes + \
            1 if self.add_background_class else self.num_classes
        self.use_dropout = desc.use_dropout if 'use_dropout' in desc else False
        self.dropout_keep_prob = desc.dropout_keep_prob if 'dropout_keep_prob' in desc else 1.0
        self.box_code_size = desc.box_code_size if 'box_code_size' in desc else 4
        self.share_box_across_classes = desc.share_box_across_classes if 'share_box_across_classes' in desc else False
        self.fc_hyperparams = scope_generator.get_hyper_params_scope(
            desc.fc_hyperparams)

    def get_real_model(self, training):
        """Get real model of maskRcnnBox."""
        if self.model:
            return self.model
        else:
            self.box_prediction_head = box_head.MaskRCNNBoxHead(
                is_training=training,
                num_classes=self.num_classes,
                fc_hyperparams_fn=self.fc_hyperparams,
                use_dropout=self.use_dropout,
                dropout_keep_prob=self.dropout_keep_prob,
                box_code_size=self.box_code_size,
                share_box_across_classes=self.share_box_across_classes)
            self.class_prediction_head = class_head.MaskRCNNClassHead(
                is_training=training,
                num_class_slots=self.num_class_slots,
                fc_hyperparams_fn=self.fc_hyperparams,
                use_dropout=self.use_dropout,
                dropout_keep_prob=self.dropout_keep_prob)

            third_stage_heads = {}
            self.model = mask_rcnn_box_predictor.MaskRCNNBoxPredictor(
                is_training=training,
                num_classes=self.num_classes,
                box_prediction_head=self.box_prediction_head,
                class_prediction_head=self.class_prediction_head,
                third_stage_heads=third_stage_heads)
            return self.model

    def __call__(self, features, labels, training):
        """Forward function of maskRcnnBox."""
        return self.get_real_model(training).predict(features, labels)
