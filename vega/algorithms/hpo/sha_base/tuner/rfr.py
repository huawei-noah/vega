# -*- coding: utf-8 -*-

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

"""RandomForestWithStdRegressor."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class RandomForestWithStdRegressor(RandomForestRegressor):
    """RandomForestWithStdRegressor."""

    def __init__(self, **kwargs):
        """Init RandomForestWithStdRegressor."""
        super(RandomForestWithStdRegressor, self).__init__(**kwargs)

    def predict(self, X):
        """Predict a mean and std for input X.

        :param X:
        :return:
        """
        X = self._validate_X_predict(X)
        t_pre = []
        for e in self.estimators_:
            _accumulate_prediction_std(e.predict, X, t_pre)
        t_pre = np.concatenate(t_pre, axis=1)
        mean = t_pre.mean(axis=1)
        std = t_pre.std(axis=1)
        return mean, std


def _accumulate_prediction_std(predict, X, out):
    """Accumulate prediction std.

    This is a utility function for joblib's Parallel.
    It can't go locally in ForestClassifier or ForestRegressor, because joblib
    complains that it cannot pickle it when placed there.
    """
    prediction = predict(X, check_input=False)
    out.append(prediction.reshape(-1, 1))
