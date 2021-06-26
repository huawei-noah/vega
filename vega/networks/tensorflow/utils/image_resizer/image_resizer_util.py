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
from object_detection.core import preprocessor


def get_image_resizer(desc):
    """Get image resizer function."""
    image_resizer_type = desc.type
    min_dimension = desc.min_dimension
    max_dimension = desc.max_dimension
    pad_to_max_dimension = desc.pad_to_max_dimension if 'pad_to_max_dimension' in desc else False
    resize_method = desc.resize_method if 'resize_method' in desc else tf.image.ResizeMethod.BILINEAR
    if image_resizer_type == 'keep_aspect_ratio_resizer':
        if not (min_dimension <= max_dimension):
            raise ValueError('min_dimension > max_dimension')
        per_channel_pad_value = (0, 0, 0)
        if 'per_channel_pad_value' in desc and desc.per_channel_pad_value:
            per_channel_pad_value = tuple(desc.per_channel_pad_value)
        image_resizer_fn = functools.partial(
            preprocessor.resize_to_range,
            min_dimension=min_dimension,
            max_dimension=max_dimension,
            method=resize_method,
            pad_to_max_dimension=pad_to_max_dimension,
            per_channel_pad_value=per_channel_pad_value)
        return image_resizer_fn
    else:
        raise ValueError(
            'Invalid image resizer option: \'%s\'.' % image_resizer_type)
