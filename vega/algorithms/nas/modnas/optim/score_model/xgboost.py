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

"""XGBoost score prediction model."""

from collections import OrderedDict
from typing import List
from numpy import ndarray
import numpy as np
try:
    import xgboost as xgb
except ImportError:
    xgb = None
from modnas.registry.score_model import register
from .base import ScoreModel


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

    def __init__(self, space, loss_type='reg', xgb_kwargs=None):
        super().__init__(space)
        if xgb_kwargs is None:
            xgb_kwargs = {}
        if xgb is None:
            raise RuntimeError('xgboost is not installed')
        xgb_params = xgb_params_rank if loss_type == 'rank' else xgb_params_reg
        xgb_params.update(xgb_kwargs)
        self.xgb_params = xgb_params
        self.model = None

    def fit(self, inputs: List[OrderedDict], results: List[float]) -> None:
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

    def predict(self, inputs: List[OrderedDict]) -> ndarray:
        """Return predicted evaluation score from model."""
        feats = self.to_feature(inputs)
        dtest = xgb.DMatrix(feats)
        return self.model.predict(dtest)
