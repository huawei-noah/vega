# -*- coding:utf-8 -*-

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

"""Quota item base."""

import vega
from vega.core.pipeline.conf import PipeStepConfig


class QuotaItemBase(object):
    """Restrict and Terminate Base Calss."""

    def get_input_data(self):
        """Get input data."""
        count_input = None
        dataset_name = PipeStepConfig.dataset.type
        dataloader = vega.get_dataset(dataset_name).loader
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
