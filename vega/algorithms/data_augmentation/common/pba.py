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

"""
Population Based Augmentation Algorithm.

https://arxiv.org/abs/1905.05393

.. code-block:: python
    Detail of PBA:
    PBA implementation:
    Step: In each iteration we run an epoch of gradient descent.
    Eval: We evaluate a trial on a validation set not used for PBT training and disjoint from
          the final test set.
    Ready: A trial is ready to go through the exploit-and-explore process once 3 steps/epochs
           have elapsed.
    Exploit: We use Truncation Selection, where a trial in the bottom 25% of the population
             clones the weights and hyperparameters of a model in the top 25%
    Explore: For each hyperparameter, we either uniformly resample from all possible values
             or perturb the original value.
"""
import operator
import shutil
import numpy as np
import pandas as pd
from vega.common import FileOps
from .status_type import StatusType


class PBA(object):
    """PBA (Population Based Augmentation).

    :param int config_count: Total config or hyperparameter count.
    :param each_epochs: number of epochs for each trainer.
    :type each_epochs: int.
    :param total_rungs: number of rungs for PBA search.
    :type total_rungs: int.
    """

    def __init__(self, config_count, each_epochs, total_rungs, local_base_path, num_operation):
        """Init PBA."""
        self.total_rungs = total_rungs
        self.each_epochs = each_epochs
        self.config_count = config_count
        self.local_base_path = local_base_path
        self.is_completed = False
        self.rung_id = 0
        self.sieve_board = pd.DataFrame(
            columns=['rung_id', 'config_id', 'status', 'score'])
        self.best_score_dict = {}
        self.all_config_dict = {}
        self.total_propose = 0
        for i in range(self.total_rungs):
            self.best_score_dict[i] = {}
        for i in range(self.config_count):
            self.all_config_dict[i] = {}
        config = [0] * num_operation * 4
        for i in range(self.config_count):
            self.all_config_dict[i][0] = config
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
            idx = np.random.randint(0, self.config_count)
            result = {'config_id': idx,
                      'score': -1 * float('inf'),
                      'configs': {'PBAconfig': self.all_config_dict[idx]}}
            return result
        else:
            for rung_id in reversed(range(0, self.total_rungs)):
                if len(self.best_score_dict[rung_id]) > 0:
                    idx = max(self.best_score_dict[rung_id].items(),
                              key=operator.itemgetter(1))[0]
                    result = {'config_id': idx,
                              'score': self.best_score_dict[rung_id][idx],
                              'configs': {'PBAconfig': self.all_config_dict[idx]}}
                    return result

    def add_score(self, config_id, rung_id, score):
        """Update the sieve_board for add score.

        :param int config_id: Description of parameter `config_id`.
        :param int rung_id: Description of parameter `rung_id`.
        :param float score: Description of parameter `score`.
        """
        _key = (self.sieve_board['config_id'] == config_id) & \
               (self.sieve_board['rung_id'] == rung_id)
        self.sieve_board.loc[_key, ['status', 'score']] = [
            StatusType.FINISHED, score]
        if rung_id > 0 and config_id not in self.best_score_dict[rung_id]:
            self.best_score_dict[rung_id][config_id] = -1 * float('inf')
        if score > self.best_score_dict[rung_id][config_id]:
            self.best_score_dict[rung_id][config_id] = score
        if self._check_rung_finished():
            # if current rung task all finished, we should init next rung
            self._init_next_rung()
        self.is_completed = self._check_completed()
        return

    def explore(self, policy):
        """Explore the policy.

        :param policy: current policy to be explored
        :type policy: list
        :return: new policy which has been explored
        :rtype: list
        """
        new_policy = []
        for i, parameter in enumerate(policy):
            if np.random.random() < 0.2:
                new_policy.append(np.random.randint(0, 10 - i % 2))
            else:
                amt = np.random.randint(0, 4)
                if np.random.random() < 0.5:
                    new_policy.append(max(0, parameter - amt))
                else:
                    new_policy.append(min(10 - i % 2, parameter + amt))
        return new_policy

    def _init_next_rung(self):
        """Init next rung to search."""
        next_rung_id = self.rung_id + 1
        if next_rung_id >= self.total_rungs:
            self.rung_id = self.rung_id + 1
            return
        for i in range(self.config_count):
            self.all_config_dict[i][next_rung_id] = self.all_config_dict[i][self.rung_id]
        current_score = []
        for i in range(self.config_count):
            current_score.append((i, self.best_score_dict[self.rung_id][i]))
        current_score.sort(key=lambda current_score: current_score[1])
        for i in range(4):
            better_id = current_score[self.config_count - 1 - i][0]
            worse_id = current_score[i][0]
            better_worker_result_path = FileOps.join_path(self.local_base_path, 'cache', 'pba',
                                                          str(better_id), 'checkpoint')
            FileOps.make_dir(better_worker_result_path)
            worse_worker_result_path = FileOps.join_path(self.local_base_path, 'cache', 'pba',
                                                         str(worse_id), 'checkpoint')
            FileOps.make_dir(worse_worker_result_path)
            shutil.rmtree(worse_worker_result_path)
            shutil.copytree(better_worker_result_path, worse_worker_result_path)
            self.all_config_dict[worse_id] = self.all_config_dict[better_id]
            policy_unchange = self.all_config_dict[worse_id][next_rung_id]
            policy_changed = self.explore(policy_unchange)
            self.all_config_dict[worse_id][next_rung_id] = policy_changed
        for id in range(self.config_count):
            self.best_score_dict[next_rung_id][id] = -1 * float('inf')
            tmp_row_data = {'config_id': id,
                            'rung_id': next_rung_id,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        self.rung_id = self.rung_id + 1

    def propose(self):
        """Propose the next config, and change the status in board.

        :return: dict of a proposed config.
        :rtype: dict, {'config_id': int,
                       'rung_id': int,
                       'configs': array,
                       'epoch': int}
        """
        _key = (self.sieve_board['rung_id'] == self.rung_id) & \
               (self.sieve_board['status'] == StatusType.WAITTING)
        rung_df = self.sieve_board.loc[_key]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': next_config_id,
            'rung_id': self.rung_id,
            'configs': self.all_config_dict[next_config_id][self.rung_id],
            'all_configs': self.all_config_dict[next_config_id],
            'epoch': int(self.each_epochs)
        }
        self._change_status(rung_id=self.rung_id,
                            config_id=next_config_id,
                            status=StatusType.RUNNING)
        self.total_propose = self.total_propose + 1
        return results

    def _add_to_board(self, one_dict):
        """Add the new record into board.

        :param dict one_dict: Description of parameter `one_dict`.
        """
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
        if current_rung_df.empty and self.rung_id >= self.total_rungs:
            return True
        else:
            return False

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
