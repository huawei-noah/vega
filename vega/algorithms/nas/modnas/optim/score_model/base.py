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

"""Evaluation score prediction model."""

from typing import List, Union
from collections import OrderedDict
import numpy as np
from numpy import ndarray


class ScoreModel():
    """Base score prediction model class."""

    def __init__(self, space):
        self.space = space

    def fit(self, inputs, results):
        """Fit model with evaluation results."""
        raise NotImplementedError

    def predict(self, inputs):
        """Return predicted evaluation score from model."""
        raise NotImplementedError

    def _process_input(self, inp: OrderedDict) -> List[int]:
        ret = []
        for n, v in inp.items():
            p = self.space.get_param(n)
            one_hot = [0] * len(p)
            one_hot[p.get_index(v)] = 1
            ret.extend(one_hot)
        return ret

    def to_feature(self, inputs: Union[OrderedDict, List[OrderedDict]]) -> ndarray:
        """Return feature variables from inputs."""
        if not isinstance(inputs, list):
            inputs = [inputs]
        return np.array([self._process_input(inp) for inp in inputs])

    def to_target(self, results: List[float]) -> ndarray:
        """Return target variables from results."""
        def to_metrics(res):
            if isinstance(res, dict):
                return list(res.values())[0]
            if isinstance(res, (tuple, list)):
                return res[0]
            return res

        return np.array([to_metrics(r) for r in results])
