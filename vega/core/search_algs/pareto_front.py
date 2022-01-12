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

"""A basic ParetoFront class, for multi-objective optimization.

you can set in yml file `pareto` part and get cfg:
pareto:
    object_count: 2        # How many objects you want to optimize, default=2.
    max_object_ids: [0, 1] # Which of those objects you want to maximize,
                           # if the index was not in this list,
                           # that object is set to be minimize in default.
if not use yml config, can use dict cfg:
cfg = {pareto: {object_count: 2, max_object_ids: [0, 1]}}

Example:
    >>> # in yml file `pareto` part set:
    >>> # pareto:
    >>> #     object_count: 2        # 2 objects: [acc, latency]
    >>> #     max_object_ids: [0]    # maximize acc, and minimize latency.
    >>> # of if not use yml config, can use dict cfg.

    >>> cfg = {pareto: {object_count: 2, max_object_ids: [0]}}
    >>> pareto_set = ParetoFront(cfg)
    >>> for id in range(100):
    >>>     hyperparams = sample_params()  # sample a hp
    >>>     pareto_front._add_to_board(id, hyperparams) # add this hp to pareto set
    >>>     score_list = evaluate_model(hyperparams)  # get model's performance [0.9, 3.2]
    >>>     pareto_set.add_pareto_score(id, score_list) # add this hp's results to pareto set
    >>> pareto_front = pareto_set.get_pareto_front() # get the pareto front
    >>> #    {id1:hp1, id2:hp2, id6:hp6}

"""

import copy
import hashlib
import logging
import json
import pandas as pd
from vega.common.pareto_front import get_pareto_index


class ParetoFront(object):
    """ParetoFront.

    :param cfg: Description of parameter `cfg`.
    :type cfg: type
    """

    def __init__(self, object_count=2, max_object_ids=None):
        """Init for ParetoFront."""
        logging.info("start init ParetoFront")
        self.sieve_columns = ['config_id', 'sha256', 'config']
        for i in range(0, object_count):
            self.sieve_columns.append("score_{}".format(i))
        self.sieve_board = pd.DataFrame(columns=self.sieve_columns)
        self.max_object_ids = []
        if isinstance(max_object_ids, list) and len(max_object_ids) > 0:
            self.max_object_ids = [x + 3 for x in max_object_ids]
        self.pareto_cols = [x + 3 for x in range(0, object_count)]
        logging.info("finished init ParetoFront")

    @property
    def size(self):
        """Get the size of current Pareto board.

        :return: The row count of the Pareto board.
        :rtype: int
        """
        return len(self.sieve_board.index)

    def get_pareto_front(self):
        """Propose the Pareto front from the board.

        :return: The row count of the Pareto board.
        :rtype: dict
        """
        pareto_list = []
        pareto_dict = {}
        pareto_board = self.sieve_board.copy()
        pareto_board = pareto_board.dropna()
        if not pareto_board.empty:
            for max_id in self.max_object_ids:
                pareto_board.iloc[:, max_id] = pareto_board.iloc[:, max_id] * -1
            col_names = [pareto_board.columns[i] for i in self.pareto_cols]
            rewards = -1 * pareto_board[col_names].values
            indexes = get_pareto_index(rewards).tolist()
            nondominated = pareto_board[indexes]

            for tmp_list in nondominated:
                for i, value in enumerate(tmp_list):
                    if i == 2:
                        pareto_list.append(copy.deepcopy(value))
                        break
                if len(tmp_list) > 2:
                    pareto_dict[tmp_list[0]] = copy.deepcopy(tmp_list[2])
        return pareto_dict

    def add_pareto_score(self, config_id, score_list):
        """Add score into pareto board.

        :param int config_id: config id in broad data frame
        :param list score_list: the score_list contain multi-object need to set
        """
        tmp_column = self.sieve_columns.copy()
        tmp_column.remove('config_id')
        tmp_column.remove('sha256')
        tmp_column.remove('config')
        self.sieve_board.loc[
            (self.sieve_board['config_id'] == config_id),
            tmp_column
        ] = score_list
        return

    def _add_to_board(self, id, config):
        """Add a config into board.

        :param one_dict: config dict like
        :type one_dict: dict, eg.{'config_id': i, 'config': dict()}
        """
        if config is None:
            return False
        config_dict = copy.deepcopy(config)
        sha256 = hashlib.sha256(json.dumps(config_dict, sort_keys=True).encode('utf-8')).hexdigest()
        found_df = self.sieve_board[self.sieve_board['sha256'].str.contains(sha256)]
        if found_df.shape[0] > 0:
            return False
        else:
            save_dict = {'config_id': id, 'sha256': sha256, 'config': config_dict}
            self.sieve_board = self.sieve_board.append(save_dict, ignore_index=True)
            return True
