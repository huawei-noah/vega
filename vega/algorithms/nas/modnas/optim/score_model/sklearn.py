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

"""Scikit-learn score prediction model."""

import importlib
from collections import OrderedDict
from typing import List
from numpy import ndarray
import numpy as np
try:
    import sklearn
except ImportError:
    sklearn = None
from modnas.registry.score_model import register
from .base import ScoreModel


@register
class SKLearnScoreModel(ScoreModel):
    """Scikit-learn score prediction model class."""

    def __init__(self, space, model_cls, module, model_kwargs=None):
        super().__init__(space)
        if model_kwargs is None:
            model_kwargs = {}
        if sklearn is None:
            raise RuntimeError('scikit-learn is not installed')
        module = importlib.import_module(module)
        model_cls = getattr(module, model_cls)
        self.model = model_cls(**model_kwargs)

    def fit(self, inputs: List[OrderedDict], results: List[float]) -> None:
        """Fit model with evaluation results."""
        x_train = self.to_feature(inputs)
        y_train = self.to_target(results)
        index = np.random.permutation(len(x_train))
        trn_x, trn_y = x_train[index], y_train[index]
        self.model.fit(trn_x, trn_y)

    def predict(self, inputs: List[OrderedDict]) -> ndarray:
        """Return predicted evaluation score from model."""
        feats = self.to_feature(inputs)
        return self.model.predict(feats)
