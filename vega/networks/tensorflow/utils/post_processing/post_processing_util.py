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
from object_detection.core import post_processing


def _score_converter_fn_with_logit_scale(tf_score_converter_fn, logit_scale=1.0):
    """Create a function to scale logits then apply a Tensorflow function."""
    def score_converter_fn(logits):
        scaled_logits = tf.multiply(
            logits, 1.0 / logit_scale, name='scale_logits')
        return tf_score_converter_fn(scaled_logits, name='convert_scores')
    score_converter_fn.__name__ = '%s_with_logit_scale' % (
        tf_score_converter_fn.__name__)
    return score_converter_fn


def _get_score_converter_fn(score_converter_type):
    if score_converter_type == 'IDENTITY':
        return _score_converter_fn_with_logit_scale(tf.identity)
    if score_converter_type == 'SIGMOID':
        return _score_converter_fn_with_logit_scale(tf.sigmoid)
    if score_converter_type == 'SOFTMAX':
        return _score_converter_fn_with_logit_scale(tf.nn.softmax)
    raise ValueError('Unknown score converter.')


def _get_non_max_suppressor_fn(desc):
    score_threshold = desc.score_threshold if 'score_threshold' in desc else 0.0
    iou_threshold = desc.iou_threshold if 'iou_threshold' in desc else 0.6
    max_detections_per_class = desc.max_detections_per_class if 'max_detections_per_class' in desc else 100
    max_total_detections = desc.max_total_detections if 'max_total_detections' in desc else 100
    use_static_shapes = desc.use_static_shapes if 'use_static_shapes' in desc else False
    use_class_agnostic_nms = desc.use_class_agnostic_nms if 'use_class_agnostic_nms' in desc else False
    max_classes_per_detection = desc.max_classes_per_detection if 'max_classes_per_detection' in desc else 1
    soft_nms_sigma = desc.soft_nms_sigma if 'soft_nms_sigma' in desc else 0.0
    use_partitioned_nms = desc.use_partitioned_nms if 'use_partitioned_nms' in desc else False
    use_combined_nms = desc.use_combined_nms if 'use_combined_nms' in desc else False
    change_coordinate_frame = desc.change_coordinate_frame if 'change_coordinate_frame' in desc else True
    if iou_threshold < 0 or iou_threshold > 1.0:
        raise ValueError('iou_threshold not in [0, 1.0].')
    if max_detections_per_class > max_total_detections:
        raise ValueError('max_detections_per_class should be no greater than '
                         'max_total_detections.')
    if soft_nms_sigma < 0.0:
        raise ValueError('soft_nms_sigma should be non-negative.')
    if use_combined_nms and use_class_agnostic_nms:
        raise ValueError('combined_nms does not support class_agnostic_nms.')

    non_max_suppressor_fn = functools.partial(
        post_processing.batch_multiclass_non_max_suppression,
        score_thresh=score_threshold,
        iou_thresh=iou_threshold,
        max_size_per_class=max_detections_per_class,
        max_total_size=max_total_detections,
        use_static_shapes=use_static_shapes,
        use_class_agnostic_nms=use_class_agnostic_nms,
        max_classes_per_detection=max_classes_per_detection,
        soft_nms_sigma=soft_nms_sigma,
        use_partitioned_nms=use_partitioned_nms,
        use_combined_nms=use_combined_nms,
        change_coordinate_frame=change_coordinate_frame)
    return non_max_suppressor_fn


def get_post_processing_fn(desc):
    """Get post processing function."""
    nms_config = desc.batch_non_max_suppression
    score_converter_type = desc.score_converter
    non_max_suppressor_fn = _get_non_max_suppressor_fn(nms_config)
    score_converter_fn = _get_score_converter_fn(score_converter_type)
    return non_max_suppressor_fn, score_converter_fn
