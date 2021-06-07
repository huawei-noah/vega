# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evaluation score prediction model."""
import numpy as np


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

    def _process_input(self, inp):
        ret = []
        for n, v in inp.items():
            p = self.space.get_param(n)
            one_hot = [0] * len(p)
            one_hot[p.get_index(v)] = 1
            ret.extend(one_hot)
        return ret

    def to_feature(self, inputs):
        """Return feature variables from inputs."""
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [self._process_input(inp) for inp in inputs]
        return np.array(inputs)

    def to_target(self, results):
        """Return target variables from results."""
        def to_metrics(res):
            if isinstance(res, dict):
                return list(res.values())[0]
            if isinstance(res, (tuple, list)):
                return res[0]
            return res

        return np.array([to_metrics(r) for r in results])
