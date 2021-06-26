# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is Pooler for Bert."""
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class Pooler(Module):
    """Pooler layer to pooling first_token from encoder."""

    def __init__(self, config):
        super(Pooler, self).__init__()
        self.dense = ops.Linear(config.hidden_size, config.hidden_size)
        self.activation = ops.Tanh()

    def call(self, hidden_states):
        """Get token and pooling."""
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output
