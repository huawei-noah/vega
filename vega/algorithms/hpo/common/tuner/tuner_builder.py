# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Base Tuner."""
import numpy as np
import logging
from vega.core.hyperparameter_space import hp2json
from .tuner_model import TunerModel
from .acquire_function import expected_improvement, thompson_sampling, minimize_pdf
from scipy.optimize import fmin, minimize
from scipy.stats import multivariate_normal
import random

LOG = logging.getLogger("vega.hpo")


class TunerBuilder(object):
    """A Base class for Tuner."""

    def __init__(self, hyperparameter_space, gridding=False, tuner='GPEI'):
        """Init TunerBuilder.

        :param hyperparameter_space: [HyperparameterSpace]
        :param gridding:
        """
        self.min_count_score = 2
        self.hyperparameter_space = hyperparameter_space
        self.hyperparameter_list = hyperparameter_space.get_hyperparameters()
        self.tuner = tuner
        self._init_model(tuner)
        self._best_score = -1 * float('inf')
        self._best_hyperparams = None
        self.grid = gridding
        self.feature_raw = None
        self.label_raw = np.array([])
        self.feature = np.array([])
        self.label = np.array([])

    def _init_model(self, tuner_model):
        """Init model by tuner_model.

        :param tuner_model:
        :return:
        """
        self.model = TunerModel(tuner_model, self.min_count_score)
        if self.model is None:
            LOG.error('Tuner model not exist, model=%s', tuner_model)

    def get_best_hyperparams(self):
        """Get best hyperparams."""
        return self._best_hyperparams, self._best_score

    def add(self, feature, label):
        """Add feature and label to train model.

        :param feature:
        :param label:
        :return:
        """
        if ('RandSearch' in self.tuner) | ('GridSearch' in self.tuner):
            LOG.info('%s not need to use add()', self.tuner)
            return
        if isinstance(feature, dict):
            feature = [feature]
            label = [label]
        if len(feature) < 1:
            LOG.warning('Function add() failed, len(feature)<1')
            return
        if len(feature) != len(label):
            raise ValueError("The input hyperparameter list length is not  "
                             "equal to the input score list length!")
        self._add_feature_and_label(feature, label)
        self.label_raw = np.append(self.label_raw, label)
        # transform hyperparameter based on its dtype
        feature_trans = np.array([], dtype=np.float64)
        if len(self.feature_raw.shape) > 1 and self.feature_raw.shape[1] > 0:
            feature_trans = self.hyperparameter_list[0].fit_transform(
                self.feature_raw[:, 0], self.label_raw
            ).astype(float).reshape(-1, 1)

            for i_feature_row in range(1, self.feature_raw.shape[1]):
                trans = self.hyperparameter_list[i_feature_row].fit_transform(
                    self.feature_raw[:, i_feature_row], self.label_raw
                ).astype(float).reshape(-1, 1)
                feature_trans = np.column_stack((feature_trans, trans))
        # fit a new model
        self.fit(feature_trans, self.label_raw)

    def _add_feature_and_label(self, feature, label):
        """Use in `add()`, Add new param-score pairs into feature_raw and label_raw.

        :param feature:
        :param label:
        """
        for i_feature in range(len(feature)):
            each = feature[i_feature]
            if label[i_feature] > self._best_score:
                self._best_score = label[i_feature]
                self._best_hyperparams = feature[i_feature]
            tmp_param_list = []
            for param in self.hyperparameter_list:
                if param.get_name() in each:
                    tmp_param_list.append(each[param.get_name()])
                else:
                    tmp_param_list.append(param._param_range[0])
            if self.feature_raw is not None:
                self.feature_raw = np.append(
                    self.feature_raw,
                    np.array([tmp_param_list], dtype=object),
                    axis=0,
                )
            else:
                self.feature_raw = np.array([tmp_param_list], dtype=object)

    def fit(self, feature, label):
        """Fit.

        :param feature: [np.array]
        :param label: [np.array]
        :return [bool] success/fail
        """
        self.feature = feature
        self.label = label
        self.model.fit(feature, label)

    def propose(self, num=1):
        """Propose hyper-parameters to json.

        :param num: int, number of hps to propose, default is 1.
        :return: list
        """
        params_list = []
        for param in self._propose(num):
            params_list.append(hp2json(param))
        return params_list

    def _propose(self, num):
        """Use the trained model to propose a set of params from HyperparameterSpace.

        :param num: int, number of hps to propose.
        :return list<HyperParameter>
        """
        params_list = []
        if self.tuner == 'GridSearch':
            params = self.hyperparameter_space.get_sample_space(gridding=True)
            LOG.info('Start to transform hyper-parameters')
            for param in params:
                param = self.hyperparameter_space.inverse_transform(param)
                # Remove duplicate hyper-parameters
                if param not in params_list:
                    params_list.append(param)
            LOG.info('Finish to griding hyper-parameters, number=%s',
                     len(params_list))
        else:
            for _ in range(num):
                parameters = self.hyperparameter_space.get_sample_space(
                    gridding=self.grid, n=1000)
                if parameters is None:
                    LOG.error(
                        'Sample space of HyperparameterSpace acquire failed, ds=%s',
                        self.hyperparameter_space.get_hyperparameter_names())
                    return None
                if self.tuner == "TPE":
                    self.mean = self.model.mean()
                    self.cov = self.model.covariance()
                    predictions = parameters
                else:
                    predictions = self.predict(parameters)
                index = self.acquire_function(predictions)
                param = self.hyperparameter_space.inverse_transform(
                    parameters[index, :])
                params_list.append(param)
        return params_list

    def predict(self, feature):
        """Predict.

        :param feature: [np.array] Array of hyperparameters
        :return: [np.array] Array of predicted scores with shape (n_samples)
        """
        return self.model.predict(feature)

    def acquire_function(self, predictions):
        """Acquire_function.

        :param predictions:
        :return:
        """
        if ('GPEI' in self.tuner):
            return expected_improvement(predictions, None)
        elif ('EI' in self.tuner) | ('SMAC' == self.tuner):
            return expected_improvement(predictions, self._best_score)
        elif ('TS' in self.tuner) | ('SMAC-P' == self.tuner):
            return thompson_sampling(self.feature, predictions)
        elif 'TPE' in self.tuner:
            return minimize_pdf(predictions, self.mean, self.cov)
        elif 'RandSearch' in self.tuner:
            LOG.info('No need to acquire function, name=%s', self.tuner)
            return np.argmax(predictions)
        else:
            LOG.error('Acquire Function not exist, name=%s', self.tuner)
        return None
