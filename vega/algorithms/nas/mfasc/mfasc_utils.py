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

"""Utilities (predictors) for multi-fidelity active search."""
import copy
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import linregress
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from . import mfgpr
from .conf import MFASCConfig


class MFModel:
    """Base class for Multifidelity inference model."""

    def __init__(self, **args):
        """Init model."""
        return

    def fit(self, X_train_lf, y_train_lf, X_train_hf, y_train_hf):
        """Fits a model to low- and high- fidelity samples."""
        raise NotImplementedError

    def predict_lf(self, X):
        """Predicts low-fidelity values."""
        raise NotImplementedError

    def predict_hf(self, X):
        """Predicts low-fidelity values."""
        raise NotImplementedError


class MFBaggingRegressorStacked(MFModel):
    """Stacked Gradient Boosting Regression predictor."""

    def __init__(self, **args):
        """Init model."""
        self.model_lf = BaggingRegressor(**copy.deepcopy(args))
        self.model_hf = BaggingRegressor(**copy.deepcopy(args))

    def fit(self, X_train_lf, y_train_lf, X_train_hf, y_train_hf):
        """Fits a model to low- and high- fidelity samples using stacking scheme for BaggingRegressor."""
        self.model_lf.fit(X_train_lf, y_train_lf)
        X_train_hf = np.hstack((X_train_hf, self.model_lf.predict(X_train_hf).reshape(-1, 1)))
        self.model_hf.fit(X_train_hf, y_train_hf)

    def predict_hf(self, X):
        """Predict low-fidelity values."""
        y_pred_lf = self.model_lf.predict(X)
        X = np.hstack((X, y_pred_lf.reshape(-1, 1)))

        base_preds = [e.predict(X) for e in self.model_hf.estimators_]

        y_pred_hf = np.mean(base_preds, axis=0)

        rho = linregress(y_pred_lf, y_pred_hf)[0]  # get slope

        return rho, y_pred_hf, np.std(base_preds, axis=0)

    def predict_lf(self, X):
        """Predict low-fidelity values."""
        base_preds = [e.predict(X) for e in self.model_lf.estimators_]

        return np.mean(base_preds, axis=0), np.std(base_preds, axis=0)


class MFGPR(MFModel):
    """Multi-fidelity Gaussian Process Regression predictor."""

    def __init__(self, **args):
        """Init model."""
        self.model = mfgpr.GaussianProcessCoKriging(**copy.deepcopy(args))

    def fit(self, X_train_lf, y_train_lf, X_train_hf, y_train_hf):
        """Fits a model to low- and high- fidelity samples using stacking scheme for BaggingRegressor."""
        self.model.fit(X_train_lf, y_train_lf, X_train_hf, y_train_hf)

    def predict_hf(self, X):
        """Predicts low-fidelity values."""
        pred_mean, pred_std = self.model.predict(X, return_std=True)

        return self.model.rho, pred_mean, pred_std

    def predict_lf(self, X):
        """Predicts low-fidelity values."""
        pred_mean, pred_std = self.model.predict(X, return_std=True)

        return pred_mean, pred_std


def make_mf_predictor(config=MFASCConfig()):
    """Make a multi-fidelity model based on config."""
    if config.predictor_type == 'gb_stacked':
        return MFBaggingRegressorStacked(base_estimator=GradientBoostingRegressor(
            n_estimators=50,
            max_depth=5,
        ),
            n_estimators=20,
            max_samples=0.51,
            n_jobs=1)
    elif config.predictor_type == 'mfgpr':
        composite_kernel = RBF(length_scale=1, length_scale_bounds=(0.001, 100))
        composite_kernel = ConstantKernel(1, constant_value_bounds=(0.001, 100)) * composite_kernel
        composite_kernel = WhiteKernel(noise_level=1, noise_level_bounds=(0.001, 100)) + composite_kernel
        return MFGPR(kernel=composite_kernel, n_restarts_optimizer=3)
    else:
        raise ValueError("Unknown name, possible options: 'xgb_stacked' and 'mfgpr'")
