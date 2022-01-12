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

"""NASBench estimator."""
from modnas.core.params import Categorical
from modnas.registry.estim import RegressionEstim
from modnas.registry.construct import register as register_constructor
from modnas.registry.estim import register as register_estim
try:
    from nasbench import api
except ImportError:
    api = None

INPUT = 'input'
OUTPUT = 'output'

CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'


@register_constructor
class NASBenchSpaceConstructor():
    """NASBench space constructor."""

    def __init__(self, n_nodes=7):
        self.n_nodes = n_nodes

    def __call__(self, model):
        """Run constructor."""
        del model
        n_nodes = self.n_nodes
        n_states = n_nodes - 2
        n_edges = n_nodes * (n_nodes - 1) // 2
        _ = [Categorical([0, 1]) for i in range(n_edges)]
        _ = [Categorical([CONV1X1, CONV3X3, MAXPOOL3X3]) for i in range(n_states)]


class NASBenchPredictor():
    """NASBench predictor."""

    def __init__(self, record_path, record_key='test_accuracy'):
        if api is None:
            raise RuntimeError('nasbench api is not installed')
        self.nasbench = api.NASBench(record_path)
        self.record_key = record_key
        self.max_nodes = 7

    def predict(self, arch_desc):
        """Return predicted evaluation results."""
        max_nodes = self.max_nodes
        matrix = [[0] * max_nodes for i in range(max_nodes)]
        g_matrix = [g for g in arch_desc if g in [0, 1]]
        g_ops = arch_desc[len(g_matrix):]
        k = 0
        for i in range(max_nodes):
            for j in range(i + 1, max_nodes):
                matrix[i][j] = int(g_matrix[k])
                k += 1
        ops = [INPUT] + g_ops + [OUTPUT]
        model_spec = api.ModelSpec(matrix=matrix, ops=ops)
        try:
            data = self.nasbench.query(model_spec)
            val_acc = data[self.record_key]
        except api.OutOfDomainError:
            val_acc = 0
        return val_acc


@register_estim
class NASBenchEstim(RegressionEstim):
    """NASBench regression estimator."""

    def run(self, optim):
        """Run Estimator routine."""
        config = self.config
        self.logger.info('loading NASBench data')
        self.predictor = NASBenchPredictor(config.record_path)
        return super().run(optim)
