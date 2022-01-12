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
