# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""
Asynchronous Successive Halving Algorithm.

https://arxiv.org/abs/1810.05934

https://blog.ml.cmu.edu/2018/12/12/massively-parallel-hyperparameter-optimization/
.. code-block:: python
    Detail of ASHA:
    Input: minimum resource r, maximum resource R, reduction factor η, minimum
        early-stopping rate s
    Algorithm: ASHA()
        repeat
            for each free worker do
                (θ; k) = get_job()
                run_then_return_val_loss(θ; rη^(s+k))
                end
            for completed job (θ, k) with loss l do
                Update configuration θ in rung k with loss l.
            end
    Procedure: get_job()
        // Check to see if there is a promotable config.
        for k = |log_η(R=r)| − s; ... ; 1; 0 do
            candidates = top_k(rung k; |rung k|=η)
            promotable = ft for t 2 candidates if t not already promotedg
            if |promotable| > 0 then
                return promotable[0]; k + 1
            end
        end
        Draw random configuration θ. // If not, grow bottom rung.
        return θ; 0
"""
import math
import random
import operator
from math import log
from .sha_base import ShaBase
from .status_type import StatusType


class ASHA(ShaBase):
    """ASHA (Asynchronous Successive Halving Algorithm).

    :param hyperparameter_space: a pre-defined search space.
    :type hyperparameter_space: object, instance os `HyperparameterSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, hyperparameter_space, config_count, max_epochs, min_epochs=1,
                 eta=3):
        """Init ASHA."""
        super().__init__(hyperparameter_space, config_count, max_epochs, min_epochs,
                         eta)
        self.s_max = int(log(max_epochs / min_epochs) / log(eta))
        self.single_epoch = min_epochs
        # minimum early-stopping rate s
        self.sr = 0
        self.total_rungs = self.s_max + 1 - self.sr

        # init all the configs
        for i in range(self.total_rungs):
            self.best_score_dict[i] = {}
        config_list = self.get_hyperparameter_space(config_count)
        for i, config in enumerate(config_list):
            self.all_config_dict[i] = config
            self.best_score_dict[0][i] = -1 * float('inf')
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
            for rung_id in reversed(range(0, self.total_rungs)):
                if len(self.best_score_dict[rung_id]) > 0:
                    idx = max(self.best_score_dict[rung_id].items(),
                              key=operator.itemgetter(1))[0]
                    result = {'config_id': idx,
                              'score': self.best_score_dict[rung_id][idx],
                              'configs': self.all_config_dict[idx]}
                    return result

    def add_score(self, config_id, rung_id, score):
        """Update the sieve_board for add score.

        :param int config_id: Description of parameter `config_id`.
        :param int rung_id: Description of parameter `rung_id`.
        :param float score: Description of parameter `score`.

        """
        self.sieve_board.loc[(self.sieve_board['config_id'] == config_id) & (
            self.sieve_board['rung_id'] == rung_id), ['status', 'score']
        ] = [StatusType.FINISHED, score]

        if rung_id > 0 and config_id not in self.best_score_dict[rung_id]:
            self.best_score_dict[rung_id][config_id] = -1 * float('inf')

        if score > self.best_score_dict[rung_id][config_id]:
            self.best_score_dict[rung_id][config_id] = score
        # the last config is best k score, propose a new config for next rung
        if rung_id == 0 and config_id == self.sieve_board['config_id'].max():
            top_k = self._get_top_k_config_ids(rung_id)
            if top_k and config_id in top_k:
                return
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
        # Check to see if there is a promotable config
        for rung_id in reversed(range(0, self.s_max - self.sr)):
            candidate_ids = self._get_top_k_config_ids(rung_id)
            if candidate_ids is not None:
                promote_rung_id = rung_id + 1
                s_epoch = self.single_epoch * math.pow(self.eta,
                                                       (promote_rung_id + self.sr))
                promote_config_id = candidate_ids[0]
                results = {
                    'config_id': promote_config_id,
                    'rung_id': promote_rung_id,
                    'configs': self.all_config_dict[promote_config_id],
                    'epoch': int(s_epoch)
                }
                self._change_status(rung_id=rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.PORMOTED)
                self._change_status(rung_id=promote_rung_id,
                                    config_id=promote_config_id,
                                    status=StatusType.RUNNING)
                self.total_propose = self.total_propose + 1
                return results

        # Draw random configuration θ from bottom rung.
        bottom_rung = 0
        _key = (self.sieve_board['rung_id'] == bottom_rung) & \
            (self.sieve_board['status'] == StatusType.WAITTING)
        rung_df = self.sieve_board.loc[_key]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': next_config_id,
            'rung_id': bottom_rung,
            'configs': self.all_config_dict[next_config_id],
            'epoch': int(self.single_epoch)
        }
        self._change_status(rung_id=bottom_rung,
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
        _key = (self.sieve_board['config_id'] == config_id) & \
            (self.sieve_board['rung_id'] == rung_id)
        change_df = self.sieve_board.loc[_key]
        if change_df.empty:
            tmp_row_data = {'rung_id': rung_id,
                            'config_id': config_id,
                            'status': status}
            self._add_to_board(tmp_row_data)
        else:
            self.sieve_board.loc[_key, ['status']] = [status]

    def _check_completed(self):
        """Check task is completed.

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

    def _get_top_k_config_ids(self, rung_id):
        """Get top k configs.

        :param int rung_id: Description of parameter `rung_id`.
        :return: a list of top k config id.
        :rtype: list or None.

        """
        _key = (self.sieve_board['rung_id'] == rung_id) & \
            (self.sieve_board['status'].isin([StatusType.FINISHED, StatusType.PORMOTED]))
        rung_df = self.sieve_board.loc[_key]
        if rung_df.empty:
            return None
        else:
            k = int(len(rung_df.index) / self.eta)
            if k <= 0:
                return None
            a = self.best_score_dict[rung_id]
            new_a = dict(sorted(a.items(), key=operator.itemgetter(1),
                                reverse=True)[:k])
            candidate_ids = new_a.keys()
            candidate_df = rung_df.loc[
                (self.sieve_board['status'] == StatusType.FINISHED) & (
                    self.sieve_board['config_id'].isin(candidate_ids))]
            if candidate_df.empty:
                return None
            else:
                return candidate_df['config_id'].tolist()
