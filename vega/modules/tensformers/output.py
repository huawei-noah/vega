# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is Output classes."""
from vega.modules.operators import ops
from vega.modules.module import Module
from vega.common.class_factory import ClassType, ClassFactory


@ClassFactory.register(ClassType.NETWORK)
class BertOutput(Module):
    """Bert Output."""

    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = ops.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = ops.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        """Call BertOutput."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@ClassFactory.register(ClassType.NETWORK)
class BertSelfOutput(Module):
    """Bert Self Output."""

    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = ops.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = ops.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = ops.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor):
        """Call Bert Self Output."""
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
