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

"""Class of Tuner model."""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from vega.algorithms.hpo.sha_base.tuner.ParzenEstimator import ParzenEstimator
from .rfr import RandomForestWithStdRegressor


class TunerModel(object):
    """Gaussian Process and Random Forest.

    :param model_name: model's name.
    :type model_name: str
    :param min_count_score: min_count_score.
    :type min_count_score: int
    """

    def __init__(self, model_name, min_count_score, hyperparameter_list):
        """Init TunerModel."""
        self.model = None
        self.model_name = model_name
        self.min_count_score = min_count_score
        if 'GP' == model_name:
            # change different kernel for Gaussian Process
            kernel = 1.0 * Matern(length_scale=1.0,
                                  length_scale_bounds=(1e-1, 10.0), nu=1.5)
            self.model = \
                make_pipeline(StandardScaler(),
                              GaussianProcessRegressor(kernel=kernel,
                                                       normalize_y=True))
        elif 'RF' == model_name:
            self.model = RandomForestWithStdRegressor(random_state=42,
                                                      n_estimators=500,
                                                      max_depth=8,
                                                      min_samples_split=2,
                                                      n_jobs=-1)
        elif 'TPE' == model_name:
            self.model = ParzenEstimator(hyperparameter_list)
        elif ('GridSearch' == model_name) | ('RandSearch' == model_name):
            self.model = True
        else:
            warnings.warn(('Cannot init model in tuner, name=%s' % model_name),
                          RuntimeWarning)

    def fit(self, feature, label):
        """Call model's fit function.

        :param feature:
        :param label:
        :return:
        """
        if feature.shape[0] < self.min_count_score:
            warnings.warn('Unable to fit of %s, feature.shape[0]<%s' % (
                self.model_name, self.min_count_score), RuntimeWarning)
            return False
        if self.model is not None:
            self.model.fit(feature, label)

    def predict(self, feature):
        """Call predict function based on model_name.

        :param feature:
        :return:
        """
        if feature.shape[0] < self.min_count_score:
            # if param-score pair count is not enough,
            # use uniform model to propose
            warnings.warn('Using RandomSearch propose params, since current '
                          'param-score pairs count is not meet '
                          '`min_count_scores` threshold.', RuntimeWarning)
            return np.random.rand(feature.shape[0], 1)

        if self.model is None:
            warnings.warn('Using RandomSearch propose params, since %s model '
                          'not been trained yet.' % self.model_name,
                          RuntimeWarning)
            return np.random.rand(feature.shape[0], 1)
        elif 'GP' == self.model_name:
            label, std = self.model.predict(feature, return_std=True)
            return np.array(list(zip(label, std)))
        elif 'RF' == self.model_name:
            label, std = self.model.predict(feature)
            return np.array(list(zip(label, std)))
        elif 'TPE' == self.model_name:
            label, std = self.model.predict(feature)
            return np.array(list(zip(label, std)))
        elif 'RandSearch' == self.model_name:
            return np.random.rand(feature.shape[0], 1)
        else:
            return None

    def mean(self):
        """Get the mean of model."""
        return self.model.means_

    def covariance(self):
        """Get the covariance of model."""
        return self.model.covariances_
