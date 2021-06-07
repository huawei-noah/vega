# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Scikit-learn score prediction model."""
import importlib
import numpy as np
try:
    import sklearn
except ImportError:
    sklearn = None
from .base import ScoreModel
from modnas.registry.score_model import register


@register
class SKLearnScoreModel(ScoreModel):
    """Scikit-learn score prediction model class."""

    def __init__(self, space, model_cls, module, model_kwargs={}):
        super().__init__(space)
        if sklearn is None:
            raise RuntimeError('scikit-learn is not installed')
        module = importlib.import_module(module)
        model_cls = getattr(module, model_cls)
        self.model = model_cls(**model_kwargs)

    def fit(self, inputs, results):
        """Fit model with evaluation results."""
        x_train = self.to_feature(inputs)
        y_train = self.to_target(results)
        index = np.random.permutation(len(x_train))
        trn_x, trn_y = x_train[index], y_train[index]
        self.model.fit(trn_x, trn_y)

    def predict(self, inputs):
        """Return predicted evaluation score from model."""
        feats = self.to_feature(inputs)
        return self.model.predict(feats)
