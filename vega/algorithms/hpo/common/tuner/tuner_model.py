# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Class of Tuner model."""
import warnings
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (Matern)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .rfr import RandomForestWithStdRegressor
from vega.algorithms.hpo.common.tuner.double_gaussian import DoubleMultiGaussian


class TunerModel(object):
    """Gaussian Process and Random Forest.

    :param model_name: model's name.
    :type model_name: str
    :param min_count_score: min_count_score.
    :type min_count_score: int
    """

    def __init__(self, model_name, min_count_score):
        """Init TunerModel."""
        self.model = None
        self.model_name = model_name
        self.min_count_score = min_count_score
        if 'GP' in model_name:
            # change different kernel for Gaussian Process
            kernel = 1.0 * Matern(length_scale=1.0,
                                  length_scale_bounds=(1e-1, 10.0), nu=1.5)
            self.model = \
                make_pipeline(StandardScaler(),
                              GaussianProcessRegressor(kernel=kernel,
                                                       normalize_y=True))
        elif 'SMAC' in model_name:
            self.model = RandomForestWithStdRegressor(random_state=42,
                                                      n_estimators=500,
                                                      max_depth=8,
                                                      min_samples_split=2,
                                                      n_jobs=-1)
        elif 'TPE' in model_name:
            self.model = DoubleMultiGaussian()
        elif ('GridSearch' in model_name) | ('RandSearch' in model_name):
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
        elif 'GP' in self.model_name:
            label, std = self.model.predict(feature, return_std=True)
            return np.array(list(zip(label, std)))
        elif 'SMAC' in self.model_name:
            label, std = self.model.predict(feature)
            return np.array(list(zip(label, std)))
        elif 'TPE' in self.model_name:
            label, std = self.model.predict(feature)
            return np.array(list(zip(label, std)))
        elif 'RandSearch' in self.model_name:
            return np.random.rand(feature.shape[0], 1)
        else:
            return None

    def mean(self):
        """Get the mean of model."""
        return self.model.means_

    def covariance(self):
        """Get the covariance of model."""
        return self.model.covariances_
