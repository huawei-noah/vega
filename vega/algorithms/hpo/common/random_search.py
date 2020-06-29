# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Class of Random (search)."""
import random
from .sha_base import ShaBase
from .status_type import StatusType


class RandomSearch(ShaBase):
    """Random Search from a given search hyperparameter_space.

    :param hyperparameter_space: a pre-defined search space.
    :type hyperparameter_space: object, instance os `HyperparameterSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int epoch: init epoch for each propose.
    """

    def __init__(self, hyperparameter_space, config_count, epoch):
        """Init random search."""
        super().__init__(hyperparameter_space, config_count, epoch, 1, 3)
        # init the config list
        self.config_list = self.get_hyperparameter_space(config_count)
        self.best_config_id = 0
        self.best_score = -1 * float('inf')
        self.epoch = epoch
        for i, config in enumerate(self.config_list):
            tmp_row_data = {'rung_id': 0,
                            'config_id': i,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        return

    def best_config(self):
        """Get current best score config.

        :return:  dict
            {'config_id': int,
                'score': float,
                'configs': dict}
            config_id, score, and configs of the current best config.
        """
        if self.total_propose == 0:
            idx = random.randint(0, len(self.config_list))
            result = {'config_id': idx,
                      'score': -1 * float('inf'),
                      'configs': self.config_list[idx]}
            return result
        else:
            result = {'config_id': self.best_config_id,
                      'score': self.best_score,
                      'configs': self.config_list[self.best_config_id]}
            return result

    def add_score(self, config_id, score):
        """Add score into best score dict and board of sha bracket.

        :param config_id: config id in broad data frame
        :param score: the best score need to set
        """
        self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id),
            ['status', 'score']
        ] = [StatusType.FINISHED, score]

        if score > self.best_score:
            self.best_config_id = config_id
            self.best_score = score

        self.is_completed = self._check_completed()
        return

    def propose(self):
        """Propose the next hyper parameter.

        :return: dict
        """
        rung_df = self.sieve_board.loc[(self.sieve_board['status'] == StatusType.WAITTING)]
        if rung_df.empty:
            return None
        next_config_id = rung_df['config_id'].min(skipna=True)
        results = {
            'config_id': int(next_config_id),
            'rung_id': 0,
            'configs': self.config_list[int(next_config_id)],
            'epoch': int(self.epoch)
        }
        self._change_status(rung_id=0,
                            config_id=next_config_id,
                            status=StatusType.RUNNING)
        self.total_propose = self.total_propose + 1
        return results

    def _check_completed(self):
        """Check all sha task completed.

        :return: True/False
        """
        current_rung_df = self.sieve_board.loc[
            self.sieve_board['status'].isin(
                [StatusType.WAITTING, StatusType.RUNNING])
        ]
        if current_rung_df.empty:
            return True
        else:
            return False
