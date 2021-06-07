# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""XGBoost score prediction model."""
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from .base import ScoreModel
from modnas.registry.score_model import register


xgb_params_reg = {
    'max_depth': 3,
    'gamma': 0.0001,
    'min_child_weight': 1,
    'subsample': 1.0,
    'eta': 0.3,
    'lambda': 1.00,
    'alpha': 0,
    'objective': 'reg:squarederror',
}

xgb_params_rank = {
    'max_depth': 3,
    'gamma': 0.0001,
    'min_child_weight': 1,
    'subsample': 1.0,
    'eta': 0.3,
    'lambda': 1.00,
    'alpha': 0,
    'objective': 'rank:pairwise',
}


@register
class XGBoostScoreModel(ScoreModel):
    """XGBoost score prediction model class."""

    def __init__(self, space, loss_type='reg', xgb_kwargs={}):
        super().__init__(space)
        if xgb is None:
            raise RuntimeError('xgboost is not installed')
        xgb_params = xgb_params_rank if loss_type == 'rank' else xgb_params_reg
        xgb_params.update(xgb_kwargs)
        self.xgb_params = xgb_params
        self.model = None

    def fit(self, inputs, results):
        """Fit model with evaluation results."""
        x_train = self.to_feature(inputs)
        y_train = self.to_target(results)
        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=400,
        )

    def predict(self, inputs):
        """Return predicted evaluation score from model."""
        feats = self.to_feature(inputs)
        dtest = xgb.DMatrix(feats)
        return self.model.predict(dtest)
