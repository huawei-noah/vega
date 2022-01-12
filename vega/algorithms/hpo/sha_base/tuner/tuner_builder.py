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

"""Base Tuner."""
import logging
import numpy as np
from .tuner_model import TunerModel
from .acquire_function import expected_improvement, thompson_sampling

LOG = logging.getLogger("vega.hpo")


class TunerBuilder(object):
    """A Base class for Tuner."""

    def __init__(self, search_space, gridding=False, tuner='GP'):
        """Init TunerBuilder.

        :param search_space: [SearchSpace]
        :param gridding:
        """
        self.min_count_score = 1
        self.search_space = search_space
        self.params = search_space.params()
        self.tuner = tuner
        self._init_model(tuner)
        self._best_score = -1 * float('inf')
        self._best_params = None
        self.grid = gridding
        self.feature_raw = None
        self.label_raw = np.array([])
        self.feature = np.array([])
        self.label = np.array([])
        self.fited = False

    def _init_model(self, tuner_model):
        """Init model by tuner_model.

        :param tuner_model:
        :return:
        """
        self.model = TunerModel(tuner_model, self.min_count_score, self.params)
        if self.model is None:
            LOG.error('Tuner model not exist, model=%s', tuner_model)

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
            feature_trans = self.params[0].encode(
                self.feature_raw[:, 0], self.label_raw
            ).astype(float).reshape(-1, 1)

            for i_feature_row in range(1, self.feature_raw.shape[1]):
                trans = self.params[i_feature_row].encode(
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
            for param in self.params:
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
        self.fited = True

    def propose(self, num=1):
        """Propose hyper-parameters to json.

        :param num: int, number of hps to propose, default is 1.
        :return: list
        """
        params_list = []
        for param in self._propose(num):
            params_list.append(dict(param))
        return params_list

    def _propose(self, num):
        """Use the trained model to propose a set of params from SearchSpace.

        :param num: int, number of hps to propose.
        :return list<HyperParameter>
        """
        params_list = []
        if self.tuner == 'GridSearch':
            params = self.search_space.get_sample_space(gridding=True)
            LOG.info('Start to transform hyper-parameters')
            for param in params:
                param = self.search_space.decode(param)
                # Remove duplicate hyper-parameters
                if param not in params_list:
                    params_list.append(param)
            LOG.info('Finish to griding hyper-parameters, number=%s',
                     len(params_list))
        else:
            if not self.fited:
                parameters = self.search_space.get_sample_space(n=num)
                if parameters is None:
                    LOG.error(
                        'Sample space of SearchSpace acquire failed, ds=%s',
                        self.search_space.get_hp_names())
                    return None
                for p in parameters:
                    params = self.search_space.decode(p)
                    params_list.append(params)
            else:
                for _ in range(num):
                    parameters = self.search_space.get_sample_space(gridding=self.grid, n=1000)
                    if parameters is None:
                        LOG.error(
                            'Sample space of SearchSpace acquire failed, ds=%s',
                            self.search_space.get_hp_names())
                        return None
                    predictions = self.predict(parameters)
                    index = self.acquire_function(predictions)
                    param = self.search_space.decode(parameters[index, :])
                    params_list.append(param)
        return params_list

    def predict(self, feature):
        """Predict.

        :param feature: [np.array] Array of hyperparameters
        :return: [np.array] Array of predicted scores with shape (n_samples)
        """
        return self.model.predict(feature)

    def acquire_function(self, predictions, method="EI"):  # EI | TS
        """Acquire_function.

        :param predictions:
        :return:
        """
        if 'RandSearch' == self.tuner:
            return np.argmax(predictions)

        if method == "EI":
            return expected_improvement(predictions, None)
        else:
            return thompson_sampling(self.feature, predictions)
