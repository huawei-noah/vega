# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Quota item base."""

import vega
from vega.core.pipeline.conf import PipeStepConfig


class QuotaItemBase(object):
    """Restrict and Terminate Base Calss."""

    def get_input_data(self):
        """Get input data."""
        count_input = None
        dataset_name = PipeStepConfig.dataset.type
        dataloader = vega.dataset(dataset_name).loader
        if vega.is_torch_backend():
            _iter = iter(dataloader)
            input_data, _ = _iter.next()
            count_input = input_data[:1]
        elif vega.is_tf_backend():
            import tensorflow as tf
            datasets = dataloader.input_fn()
            data_iter = tf.compat.v1.data.make_one_shot_iterator(datasets)
            input_data, _ = data_iter.get_next()
            count_input = input_data[:1]
        elif vega.is_ms_backend():
            data_iter = dataloader.create_dict_iterator()
            for batch in data_iter:
                count_input = batch['image']
                break
        return count_input
