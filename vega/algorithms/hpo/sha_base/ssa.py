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


r"""
Sub-Sampling.

.. code-block:: python
    Detail SSA:
    Input: the set of configuration C = {c1, ..., ck}; maximum budget R; minimum budget b; ratio eta(default eta = 3).
    output: {c_p1, ..., c_pN} with corresponding evaluations.
    1.r = 1, evaluate all configurations with budget b.
    2.for r = 2, 3, ..., [log(R / b) / log eta] do
    3.    select the leader c_l, which has the most observations.
    4.    I' = {k: c_k belongs to Cc_l, c_k <= c_l}.
    5.    if I' is None then
    6.        evaluate c_l with budget b*pow(eta, r).
    7.    else
    8.        evaluate c_k with budget b*pow(eta, r) for each k belongs to I'.
    9.    end if
    10.end for
"""

import operator
import math
from math import log
import random
from .sha_base import ShaBase
from .status_type import StatusType


class SSA(ShaBase):
    """SSA (Successive Halving Algorithm).

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int config_count: Description of parameter `config_count`.
    :param int max_epochs: Description of parameter `max_epochs`.
    :param min_epochs: Description of parameter `min_epochs`.
    :type min_epochs: int, default is 1.
    :param eta: Description of parameter `eta`.
    :type eta: int, default is 3.
    :param bool empty: default `False`.
    """

    def __init__(self, search_space, config_count, max_epochs, min_epochs=1,
                 eta=3, empty=False):
        """Init SSA."""
        super().__init__(search_space, config_count, max_epochs, min_epochs, eta)
        # hyperband algorithm init params
        self.start_id = 0
        self.s_max = int(log(max_epochs / min_epochs) / log(eta))
        self.single_epoch = min_epochs
        self.sr = 0  # minimum early-stopping rate s
        self.total_rungs = self.s_max + 1 - self.sr
        cn = int(math.sqrt(math.log(config_count * eta / (eta - 1))))
        if cn != 1:
            for i in range(cn - 1):
                self.min_epochs *= eta
        self.budget_list = []
        self.config_list = []
        current_config_num = config_count
        current_epoch = self.single_epoch
        while (current_config_num != 0):
            self.config_list.append(current_config_num)
            self.budget_list.append(min(max(current_epoch, self.min_epochs), max_epochs))
            current_config_num //= eta
            current_epoch *= eta

        if not empty:
            hyperparameter_list = self.get_hyperparameters(config_count)
            for i in range(self.total_rungs):
                self.best_score_dict[i] = {}
            for i in range(0, len(hyperparameter_list)):
                self.all_config_dict[i] = hyperparameter_list[i]
                self.best_score_dict[0][i] = -1 * float('inf')
                tmp_row_data = {'rung_id': 0,
                                'config_id': i,
                                'status': StatusType.WAITTING}
                self._add_to_board(tmp_row_data)
        return

    def add_score(self, config_id, rung_id, score):
        """Update the sieve_board for add score.

        :param int config_id: config id in board dataframe
        :param int rung_id: current rung
        :param float score: score from evaluation function of this config

        """
        self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (self.sieve_board['rung_id'] == rung_id),
                             ['status', 'score']] = [StatusType.FINISHED, score]
        if score > self.best_score_dict[rung_id][config_id]:
            self.best_score_dict[rung_id][config_id] = score
        if self._check_rung_finished():
            # if current rung task all finished, we should init next rung
            self._init_next_rung()
        self.is_completed = self._check_completed()
        return

    def propose(self):
        """Propose the next config, and change the status in board.

        :return: dict of a proposed config.
        :rtype: dict, {'config_id': int,
                       'rung_id': int,
                       'configs': dict,
                       'epoch': int}

        """
        rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == self.rung_id) & (
            self.sieve_board['status'] == StatusType.WAITTING)]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': next_config_id,
            'rung_id': self.rung_id,
            'configs': self.all_config_dict[next_config_id],
            'epoch': int(self.single_epoch)
        }
        self._change_status(rung_id=self.rung_id,
                            config_id=next_config_id,
                            status=StatusType.RUNNING)
        self.total_propose = self.total_propose + 1
        return results

    def best_config(self):
        """Get config_id, score, and configs of the current best config.

        :return: the current best config dict.
        :rtype: dict, {'config_id': int,
                        'score': float,
                        'configs': dict}

        """
        if self.total_propose == 0:
            idx = random.randint(0, len(self.all_config_dict))
            result = {'config_id': idx,
                      'score': -1 * float('inf'),
                      'configs': self.all_config_dict[idx]}
            return result
        else:
            for rung_id in reversed(range(0, self.total_rungs)):
                if len(self.best_score_dict[rung_id]) > 0:
                    idx = max(self.best_score_dict[rung_id].items(),
                              key=operator.itemgetter(1))[0]
                    result = {'config_id': idx,
                              'score': self.best_score_dict[rung_id][idx],
                              'configs': self.all_config_dict[idx]}
                    return result

    def _get_top_k_config_ids(self, k):
        """Get top k configs in current rung.

        :param int k: Description of parameter `k`.
        :return: a list of top k config id.
        :rtype: list or None.

        """
        leader_idx = max(
            self.best_score_dict[self.rung_id].items(), key=operator.itemgetter(1))[0]
        leader_score_df = self.sieve_board.loc[self.sieve_board['config_id'] == leader_idx]
        score_list = []
        for idx in range(self.config_count):
            current_id = idx + self.start_id
            if current_id == leader_idx:
                continue
            score_df = self.sieve_board.loc[self.sieve_board['config_id'] == current_id]
            rung_max = score_df['rung_id'].max(skipna=True)
            current_score = 0
            leader_score = 0
            while (rung_max > 0):
                current_score = score_df[score_df['rung_id'] == rung_max]
                leader_score = leader_score_df[leader_score_df['rung_id'] == rung_max]
                if current_score.empty or leader_score.empty:
                    rung_max = rung_max - 1
                    current_score = 0
                    leader_score = 0
                else:
                    current_score = current_score['score'].values
                    leader_score = leader_score['score'].values
                    break
            sub_score = current_score - leader_score
            score_list.append((current_id, sub_score))
        sort_list = sorted(score_list, key=operator.itemgetter(1), reverse=True)
        top_k_config = []
        top_k_config.append(leader_idx)
        if k == 1:
            return top_k_config
        for i in range(k - 1):
            top_k_config.append(sort_list[i][0])
        return top_k_config

    def _init_next_rung(self):
        """Init next rung.

        we will choose top k config from best_score_dict, and init
        config_id, rung_id, status of each row for the next rung
        update epoch with min_ephoches * (eta)^η
        """
        next_rung_id = self.rung_id + 1
        if next_rung_id >= len(self.budget_list):
            return
        # get next rung count n_i = n_(i-1) * eta^(-i)
        n_i = self.config_list[next_rung_id]
        topk_config_id = self._get_top_k_config_ids(n_i)
        self.best_score_dict[next_rung_id] = {}
        for id in topk_config_id:
            self.best_score_dict[next_rung_id][id] = -1 * float('inf')
            tmp_row_data = {'config_id': id,
                            'rung_id': next_rung_id,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        self.rung_id = self.rung_id + 1
        self.single_epoch = self.budget_list[next_rung_id]

    def set_config_list(self, config_list, start_id=0):
        """Init conifgs and add all config into board.

        :param config_list: config list which restore all hyper parameters.
        :param start_id: the index of config dict start from, default 0.
        :return: if set config list success.
        :rtype: bool.
        """
        self.start_id = start_id
        if len(config_list) != self.config_count:
            raise ValueError(
                'set config count not equal to length of config_list!')
        for i in range(self.total_rungs):
            self.best_score_dict[i] = {}
        for i in range(0, len(config_list)):
            self.all_config_dict[i + start_id] = config_list[i]
            self.best_score_dict[0][i + start_id] = -1 * float('inf')
            tmp_row_data = {'rung_id': 0,
                            'config_id': i + start_id,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        return True

    def _check_completed(self):
        """Check task is completed.

        all rows in board is not RUNNTIN or WATTING, this task is completed

        :return: if the search algorithm is finished.
        :rtype: bool.

        """
        current_rung_df = self.sieve_board.loc[
            self.sieve_board['status'].isin(
                [StatusType.WAITTING, StatusType.RUNNING])
        ]
        if current_rung_df.empty:
            return True
        else:
            return False
