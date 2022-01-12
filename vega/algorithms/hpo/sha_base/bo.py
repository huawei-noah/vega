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

"""Bayesian Optimization framework."""
import random
import operator
from .sha_base import ShaBase
from .status_type import StatusType
from .tuner import TunerBuilder


class BO(ShaBase):
    """A Bayesian Optimization framework.

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param warmup_count: `warmup_count` is the random sample count, to warm up bo alg.
    :type warmup_count: int, default is 10.
    :param alg_name: detail alg name used to propose hp.
    :type alg_name: string, ('RF', 'GP'), default is 'RF'.
    """

    def __init__(self, search_space, config_count, max_epochs, warmup_count=10,
                 alg_name='RF'):
        """Init BO."""
        super().__init__(search_space, config_count, max_epochs, 1, 3)

        # init all the configs
        self.warmup_count = warmup_count
        self.best_score = -1 * float('inf')
        self.tuner = TunerBuilder(search_space=search_space, tuner=alg_name)
        config_list = self.get_hyperparameters(self.warmup_count)
        for i, config in enumerate(config_list):
            self.all_config_dict[i] = config
            self.best_score_dict[i] = -1 * float('inf')
            tmp_row_data = {'rung_id': 0,
                            'config_id': i,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)

        return

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
            if len(self.best_score_dict) > 0:
                idx = max(self.best_score_dict.items(),
                          key=operator.itemgetter(1))[0]
                result = {'config_id': idx,
                          'score': self.best_score_dict[idx],
                          'configs': self.all_config_dict[idx]}
                return result

    def add_score(self, config_id, score):
        """Update the sieve_board for add score.

        :param int config_id: Description of parameter `config_id`.
        :param float score: Description of parameter `score`.

        """
        self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id),
            ['status', 'score']
        ] = [StatusType.FINISHED, score]

        if score > self.best_score:
            self.best_config_id = config_id
            self.best_score = score
        rung_id = 0
        self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (
            self.sieve_board['rung_id'] == rung_id), ['status', 'score']] = [StatusType.FINISHED, score]

        if config_id not in self.best_score_dict:
            self.best_score_dict[config_id] = -1 * float('inf')

        if score > self.best_score_dict[config_id]:
            self.best_score_dict[config_id] = score
        #
        if config_id in self.all_config_dict:
            # add this (config, score) pair into HP
            x = self.all_config_dict[config_id]
            score_df = self.sieve_board[self.sieve_board['config_id'] == config_id]
            score_df = score_df[score_df['rung_id'] == rung_id]
            if not score_df.empty:
                y = float(score_df['score'])
                self.tuner.add(x, y)

        current_waiting_df = self.sieve_board.loc[
            self.sieve_board['status'].isin([StatusType.WAITTING])
        ]
        if current_waiting_df.empty and self.total_propose < self.config_count:
            # get a new propose from HP
            configs = self.tuner.propose()
            config_id = len(self.all_config_dict)
            tmp_row_data = {'rung_id': 0,
                            'config_id': config_id,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
            if isinstance(configs, list):
                self.all_config_dict[config_id] = configs[0]
            else:
                self.all_config_dict[config_id] = configs
        self.is_completed = self._check_completed()
        return

    def propose(self):
        """Propose next hyper parameter.

        :return: dict of a proposed config.
        :rtype: dict, {'config_id': int,
                       'rung_id': int,
                       'configs': dict,
                       'epoch': int}

        """
        rung_df = self.sieve_board.loc[(self.sieve_board['rung_id'] == 0) & (
            self.sieve_board['status'] == StatusType.WAITTING)]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': next_config_id,
            'configs': self.all_config_dict[next_config_id],
            'epoch': int(self.max_epochs)
        }
        self._change_status(rung_id=0,
                            config_id=next_config_id,
                            status=StatusType.RUNNING)
        self.total_propose = self.total_propose + 1
        return results

    def _add_to_board(self, one_dict):
        """Add the new record into board.

        :param dict one_dict: Description of parameter `one_dict`.

        """
        # TODO pandas use contact replace append function
        self.sieve_board = self.sieve_board.append(one_dict, ignore_index=True)

    def _change_status(self, rung_id, config_id, status):
        """Change the status of each config.

        :param int rung_id: Description of parameter `rung_id`.
        :param int config_id: Description of parameter `config_id`.
        :param type status: Description of parameter `status`.
        :type enum: StatusType

        """
        change_df = self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id) & (self.sieve_board['rung_id'] == rung_id)]
        if change_df.empty:
            tmp_row_data = {'rung_id': rung_id,
                            'config_id': config_id,
                            'status': status}
            self._add_to_board(tmp_row_data)
        else:
            self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (
                self.sieve_board['rung_id'] == rung_id), ['status']] = [status]

    def _check_completed(self):
        """Check task is completed.

        :return: if the search algorithm is finished.
        :rtype: bool.

        """
        current_rung_df = self.sieve_board.loc[
            self.sieve_board['status'].isin(
                [StatusType.WAITTING, StatusType.RUNNING])
        ]
        if current_rung_df.empty and self.total_propose >= self.config_count:
            return True
        else:
            return False
