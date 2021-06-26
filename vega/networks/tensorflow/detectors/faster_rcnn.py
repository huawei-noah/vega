# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined faster rcnn detector."""
import functools
import tensorflow as tf

from object_detection.core import balanced_positive_negative_sampler as sampler
from object_detection.core import losses
from object_detection.core import post_processing
from object_detection.core import standard_fields as fields
from object_detection.core import target_assigner
from object_detection.utils import spatial_transform_ops as spatial_ops
from object_detection.meta_architectures import faster_rcnn_meta_arch

from vega.common import ClassType, ClassFactory
from vega.networks.network_desc import NetworkDesc
from vega.networks.tensorflow.utils.hyperparams import scope_generator
from vega.networks.tensorflow.utils.image_resizer import image_resizer_util
from vega.networks.tensorflow.utils.post_processing import post_processing_util


@ClassFactory.register(ClassType.NETWORK)
class FasterRCNN(object):
    """Faster RCNN."""

    def __init__(self, desc):
        """Init faster rcnn.

        :param desc: config dict
        """
        super(FasterRCNN, self).__init__()

        self.num_classes = int(desc.num_classes)
        self.number_of_stages = int(desc.number_of_stages)

        # Backbone for feature extractor
        self.feature_extractor = NetworkDesc(desc.backbone).to_model()

        # First stage anchor generator
        self.first_stage_anchor_generator = NetworkDesc(
            desc["first_stage_anchor_generator"]).to_model()

        # First stage target assigner
        self.use_matmul_gather_in_matcher = False  # Default
        self.first_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'proposal', use_matmul_gather=self.use_matmul_gather_in_matcher)

        # First stage box predictor
        self.first_stage_box_predictor_arg_scope_fn = scope_generator.get_hyper_params_scope(
            desc.first_stage_box_predictor_conv_hyperparams)
        self.first_stage_atrous_rate = 1  # Default: 1
        self.first_stage_box_predictor_kernel_size = 3  # Default
        self.first_stage_box_predictor_depth = 512  # Default
        self.first_stage_minibatch_size = 256  # Default

        # First stage sampler
        self.first_stage_positive_balance_fraction = 0.5  # Default
        self.use_static_balanced_label_sampler = False  # Default
        self.use_static_shapes = False  # Default
        self.first_stage_sampler = sampler.BalancedPositiveNegativeSampler(
            positive_fraction=self.first_stage_positive_balance_fraction,
            is_static=(self.use_static_balanced_label_sampler and self.use_static_shapes))

        # First stage NMS
        self.first_stage_nms_score_threshold = 0.0
        self.first_stage_nms_iou_threshold = 0.7
        self.first_stage_max_proposals = 300
        self.use_partitioned_nms_in_first_stage = True  # Default
        self.use_combined_nms_in_first_stage = False  # Default
        self.first_stage_non_max_suppression_fn = functools.partial(
            post_processing.batch_multiclass_non_max_suppression,
            score_thresh=self.first_stage_nms_score_threshold,
            iou_thresh=self.first_stage_nms_iou_threshold,
            max_size_per_class=self.first_stage_max_proposals,
            max_total_size=self.first_stage_max_proposals,
            use_static_shapes=self.use_static_shapes,
            use_partitioned_nms=self.use_partitioned_nms_in_first_stage,
            use_combined_nms=self.use_combined_nms_in_first_stage)

        # First stage localization loss weight
        self.first_stage_localization_loss_weight = 2.0

        # First stage objectness loss weight
        self.first_stage_objectness_loss_weight = 1.0

        # Second stage target assigner
        self.second_stage_target_assigner = target_assigner.create_target_assigner(
            'FasterRCNN', 'detection', use_matmul_gather=self.use_matmul_gather_in_matcher)

        # Second stage sampler
        self.second_stage_batch_size = 64  # Default
        self.second_stage_balance_fraction = 0.25  # Default
        self.second_stage_sampler = sampler.BalancedPositiveNegativeSampler(
            positive_fraction=self.second_stage_balance_fraction,
            is_static=(self.use_static_balanced_label_sampler and self.use_static_shapes))

        # Second stage box predictor
        self.second_stage_box_predictor = NetworkDesc(
            desc.mask_rcnn_box).to_model()

        # Second stage NMS function
        self.second_stage_non_max_suppression_fn, self.second_stage_score_conversion_fn = \
            post_processing_util.get_post_processing_fn(desc.second_stage_post_processing)

        # Second stage mask prediction loss weight
        self.second_stage_mask_prediction_loss_weight = 1.0  # default

        # Second stage localization loss weight
        self.second_stage_localization_loss_weight = 2.0

        # Second stage classification loss weight
        self.second_stage_classification_loss_weight = 1.0

        # Second stage classification loss
        self.logit_scale = 1.0  # Default
        self.second_stage_classification_loss = losses.WeightedSoftmaxClassificationLoss(
            logit_scale=self.logit_scale)

        self.hard_example_miner = None
        self.add_summaries = True

        # Crop and resize function
        self.use_matmul_crop_and_resize = False  # Default
        self.crop_and_resize_fn = (
            spatial_ops.multilevel_matmul_crop_and_resize
            if self.use_matmul_crop_and_resize
            else spatial_ops.native_crop_and_resize)

        self.clip_anchors_to_image = False  # Default
        self.resize_masks = True  # Default
        self.return_raw_detections_during_predict = False  # Default
        self.output_final_box_features = False  # Default

        # Image resizer function
        self.image_resizer_fn = image_resizer_util.get_image_resizer(
            desc.image_resizer)

        self.initial_crop_size = 14
        self.maxpool_kernel_size = 2
        self.maxpool_stride = 2

        # Real model to be called
        self.model = None

    def _init_model(self, training):

        # Init FasterRCNNMetaArch
        common_kwargs = {
            'is_training': training,
            'num_classes': self.num_classes,
            'image_resizer_fn': self.image_resizer_fn,
            'feature_extractor': self.feature_extractor.get_real_model(training),
            'number_of_stages': self.number_of_stages,
            'first_stage_anchor_generator': self.first_stage_anchor_generator.get_real_model(training),
            'first_stage_target_assigner': self.first_stage_target_assigner,
            'first_stage_atrous_rate': self.first_stage_atrous_rate,
            'first_stage_box_predictor_arg_scope_fn': self.first_stage_box_predictor_arg_scope_fn,
            'first_stage_box_predictor_kernel_size': self.first_stage_box_predictor_kernel_size,
            'first_stage_box_predictor_depth': self.first_stage_box_predictor_depth,
            'first_stage_minibatch_size': self.first_stage_minibatch_size,
            'first_stage_sampler': self.first_stage_sampler,
            'first_stage_non_max_suppression_fn': self.first_stage_non_max_suppression_fn,
            'first_stage_max_proposals': self.first_stage_max_proposals,
            'first_stage_localization_loss_weight': self.first_stage_localization_loss_weight,
            'first_stage_objectness_loss_weight': self.first_stage_objectness_loss_weight,
            'second_stage_target_assigner': self.second_stage_target_assigner,
            'second_stage_batch_size': self.second_stage_batch_size,
            'second_stage_sampler': self.second_stage_sampler,
            'second_stage_non_max_suppression_fn': self.second_stage_non_max_suppression_fn,
            'second_stage_score_conversion_fn': self.second_stage_score_conversion_fn,
            'second_stage_localization_loss_weight': self.second_stage_localization_loss_weight,
            'second_stage_classification_loss': self.second_stage_classification_loss,
            'second_stage_classification_loss_weight': self.second_stage_classification_loss_weight,
            'hard_example_miner': self.hard_example_miner,
            'add_summaries': self.add_summaries,
            'crop_and_resize_fn': self.crop_and_resize_fn,
            'clip_anchors_to_image': self.clip_anchors_to_image,
            'use_static_shapes': self.use_static_shapes,
            'resize_masks': self.resize_masks,
            'return_raw_detections_during_predict': self.return_raw_detections_during_predict,
            'output_final_box_features': self.output_final_box_features
        }

        self.model = faster_rcnn_meta_arch.FasterRCNNMetaArch(
            initial_crop_size=self.initial_crop_size,
            maxpool_kernel_size=self.maxpool_kernel_size,
            maxpool_stride=self.maxpool_stride,
            second_stage_mask_rcnn_box_predictor=self.second_stage_box_predictor.get_real_model(
                training),
            second_stage_mask_prediction_loss_weight=(
                self.second_stage_mask_prediction_loss_weight),
            **common_kwargs)

    def get_real_model(self, training):
        """Get or init real model."""
        if self.model:
            return self.model
        else:
            self._init_model(training)
            return self.model

    def __call__(self, features, labels, training):
        """Forward function of faster-rcnn."""
        if training:
            self.get_real_model(training).provide_groundtruth(
                groundtruth_boxes_list=tf.unstack(
                    labels[fields.InputDataFields.groundtruth_boxes]),
                groundtruth_classes_list=tf.unstack(
                    labels[fields.InputDataFields.groundtruth_classes]),
                groundtruth_weights_list=tf.unstack(labels[fields.InputDataFields.groundtruth_weights]))

        predict_results = self.get_real_model(training).predict(features[fields.InputDataFields.image],
                                                                features[fields.InputDataFields.true_image_shape])
        return predict_results

    def loss(self, predict_results, true_image_shapes):
        """Get loss function of faster-rcnn."""
        return self.get_real_model(True).loss(predict_results, true_image_shapes)

    def updates(self):
        """Update faster-rcnn model."""
        return self.get_real_model(True).updates()

    def regularization_losses(self):
        """Get regularization loss of faster-rcnn."""
        return self.get_real_model(True).regularization_losses()

    def restore_map(self, fine_tune_checkpoint_type, load_all_detection_checkpoint_vars):
        """Restore map of faster-rcnn."""
        return self.get_real_model(True).restore_map(
            fine_tune_checkpoint_type=fine_tune_checkpoint_type,
            load_all_detection_checkpoint_vars=(load_all_detection_checkpoint_vars))
