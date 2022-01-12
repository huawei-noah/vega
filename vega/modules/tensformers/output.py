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
