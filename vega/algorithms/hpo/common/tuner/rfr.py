# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
