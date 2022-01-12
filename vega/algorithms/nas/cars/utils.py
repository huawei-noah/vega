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

"""Util functions."""
import numpy as np
import vega


class AverageMeter(object):
    """This is a meter class to calculate average values."""

    def __init__(self):
        """Construct method."""
        self.reset()

    def reset(self):
        """Reset the meter."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the meter."""
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def eval_model_parameters(model):
    """Calculate number of parameters in million (M) for a model.

    :param model: A model
    :type model: nn.Module
    :return: The number of parameters
    :rtype: Float
    """
    if vega.is_torch_backend():
        return np.sum(v.numel() for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6
    elif vega.is_tf_backend():
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        dummy_input = tf.compat.v1.placeholder(
            dtype=tf.float32,
            shape=[1, 32, 32, 3] if model.data_format == 'channels_last' else [1, 3, 32, 32])
        model.training = True
        model(dummy_input)
        all_weight = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        weight_op = [t for t in all_weight if "auxiliary" not in t.name]
        return np.sum([np.prod(t.get_shape().as_list()) for t in weight_op]) * 1e-6
    elif vega.is_ms_backend():
        return 0
