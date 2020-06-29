# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Class of RandomPareto (search)."""
import random
import pandas as pd
import pareto
from .sha_base import ShaBase
from .status_type import StatusType


class RandomPareto(ShaBase):
    """Random Pareto Search from a given search hyperparameter_space.

    :param hyperparameter_space: a pre-defined search space.
    :type hyperparameter_space: object, instance os `HyperparameterSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int epoch: init epoch for each propose.
    :param int object_count: pareto objects count, default is `2`.
    :param list object_count: list of object that need maximum, otherwise all
        objects use minimum. default is empty.
    """

    def __init__(self, hyperparameter_space, config_count, epoch, object_count=2, max_object_ids=[]):
        """Init random pareto search."""
        super().__init__(hyperparameter_space, config_count, epoch, 1, 3)
        self.sieve_columns = ['rung_id', 'config_id', 'status']
        for i in range(0, object_count):
            self.sieve_columns.append("score_{}".format(i))
        self.sieve_board = pd.DataFrame(columns=self.sieve_columns)
        self.max_object_ids = None
        if isinstance(max_object_ids, list) and len(max_object_ids) > 0:
            self.max_object_ids = [x + 3 for x in max_object_ids]
        self.pareto_cols = [x + 3 for x in range(0, object_count)]
        # init the config list
        self.config_list = self.get_hyperparameter_space(config_count)
        self.epoch = epoch
        for i, config in enumerate(self.config_list):
            tmp_row_data = {'rung_id': 0,
                            'config_id': i,
                            'status': StatusType.WAITTING}
            self._add_to_board(tmp_row_data)
        return

    def best_config(self):
        """Get current config list located in pareto front.

        :return:  list of dict
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
            return [result]
        else:
            pareto_board = self.sieve_board.copy()
            pareto_board = pareto_board.dropna()
            nondominated = pareto.eps_sort([list(pareto_board.itertuples(False))],
                                           objectives=self.pareto_cols,
                                           epsilons=None,
                                           maximize=self.max_object_ids)
            pareto_list = []
            for tmp_list in nondominated:
                result = {}
                for i, value in enumerate(tmp_list):
                    if i == 1:
                        result['config_id'] = value
                        result['configs'] = self.config_list[int(value)]
                    elif i >= 3:
                        result[self.sieve_columns[i]] = value
                pareto_list.append(result)
            return pareto_list

    def add_score(self, config_id, score_list):
        """Add score into board of sha bracket.

        :param int config_id: config id in broad data frame
        :param list score_list: the score_list contain multi-object need to set
        """
        tmp_column = self.sieve_columns.copy()
        tmp_column.remove('rung_id')
        tmp_column.remove('config_id')
        self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id),
            tmp_column
        ] = [StatusType.FINISHED] + score_list

        self.is_completed = self._check_completed()
        return

    def propose(self):
        """Propose the next hyper parameter.

        :return: dict
        """
        rung_df = self.sieve_board.loc[
            (self.sieve_board['status'] == StatusType.WAITTING)
        ]
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
        """All sha task completed.

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
