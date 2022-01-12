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

"""ShaBase class."""
import numpy as np
import pandas as pd
from .status_type import StatusType


class ShaBase(object):
    """A Base class for successive halving and hyperband algorithm.

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int config_count: Description of parameter `config_count`.
    :param int max_epochs: Description of parameter `max_epochs`.
    :param int min_epochs: Description of parameter `min_epochs`.
    :param int eta: Description of parameter `eta`.
    """

    def __init__(self, search_space, config_count, max_epochs, min_epochs, eta):
        """Init for ShaBase class."""
        # hyperband algorithm init params
        self.eta = eta
        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.config_count = config_count
        self.is_completed = False
        self.rung_id = 0
        # init all the configs
        self.current_iter = 0
        self.search_space = search_space
        self.hyperparameter_list = self.get_hyperparameters(config_count)
        self.sieve_board = pd.DataFrame(
            columns=['rung_id', 'config_id', 'status', 'score'])
        self.config_dict = {}
        self.best_score_dict = {}
        self.all_config_dict = {}
        self.total_propose = 0

    def get_hyperparameters(self, num):
        """Use the trained model to propose a set of params from SearchSpace.

        :param int num: number of random samples from hyperparameter space.
        :return: list of random sampled config from hyperparameter space.
        :rtype: list.

        """
        params_list = []
        for i in range(num):
            parameters = self.search_space.get_sample_space()
            if parameters is None:
                return None
            predictions = np.random.rand(parameters.shape[0], 1)
            index = np.argmax(predictions)
            param = self.search_space.decode(parameters[index, :])
            params_list.append(param)
        return params_list

    def propose(self):
        """Propose the next hyper parameter.

        :raise NotImplementedError
        """
        raise NotImplementedError

    def add_score(self, config_id, rung_id, score):
        """Add score to board.

        :param int config_id: config id in board dataframe
        :param int rung_id: current rung
        :param float score: score from evaluation function of this config
        :raise NotImplementedError
        """
        raise NotImplementedError

    def _check_rung_finished(self):
        """Check if this rung is finished by status not WATTING and RUNNING.

        :return: if this rung finished.
        :rtype: bool.
        """
        current_rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == self.rung_id) & (
            self.sieve_board['status'].isin([StatusType.WAITTING, StatusType.RUNNING]))]
        if current_rung_df.empty:
            return True
        else:
            return False

    def _add_to_board(self, one_dict):
        """Add a dict into board.

        :param one_dict: config dict like
        :type one_dict: dict, eg.{'rung_id': 0, 'config_id': i, 'status': StatusType.WAITTING}

        """
        self.sieve_board = self.sieve_board.append(one_dict, ignore_index=True)

    def _change_status(self, rung_id, config_id, status):
        """Change status in board by config id and rung id.

        :param int rung_id: current rung need to update
        :param int config_id: current config id need to update
        :param enum status: status from StatusType

        """
        change_df = self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id) & (self.sieve_board['rung_id'] == rung_id)]
        if change_df.empty:
            tmp_row_data = {'rung_id': rung_id,
                            'config_id': config_id,
                            'status': status}
            self._add_to_board(tmp_row_data)
        else:
            self.sieve_board.loc[
                (self.sieve_board['config_id'] == config_id) & (self.sieve_board['rung_id'] == rung_id), ['status']] = [
                status]
