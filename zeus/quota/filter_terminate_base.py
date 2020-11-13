# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Flops and Parameters Filter."""
import copy
import zeus
from zeus.common import ClassFactory, ClassType
from zeus.common.general import General


@ClassFactory.register(ClassType.QUOTA)
class FilterTerminateBase(object):
    """Restrict and Terminate Base Calss."""

    def __init__(self):
        self.restrict_config = copy.deepcopy(General.quota.restrict)
        self.target_config = copy.deepcopy(General.quota.target)

    def is_halted(self, *args, **kwargs):
        """Decide to halt or not."""

    def is_filtered(self, desc=None):
        """Decide to filter or not."""

    def get_model_input(self, desc):
        """Get model and input."""
        from zeus.networks.network_desc import NetworkDesc
        model = NetworkDesc(desc).to_model()
        count_input = self.get_input_data()
        return model, count_input

    def get_input_data(self):
        """Get input data."""
        count_input = None
        if zeus.is_torch_backend():
            data_iter = iter(self.dataloader)
            input_data, _ = data_iter.next()
            count_input = input_data[:1]
        elif zeus.is_tf_backend():
            import tensorflow as tf
            datasets = self.dataloader.input_fn()
            data_iter = tf.compat.v1.data.make_one_shot_iterator(datasets)
            input_data, _ = data_iter.get_next()
            count_input = input_data[:1]
        elif zeus.is_ms_backend():
            data_iter = self.dataloader.create_dict_iterator()
            for batch in data_iter:
                count_input = batch['image']
                break
        return count_input
