# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
import numpy as np
import pandas as pd
import pareto
import copy
import hashlib
import json
import logging


class ParetoFront(object):
    """ParetoFront.

    :param cfg: Description of parameter `cfg`.
    :type cfg: type
    """

    def __init__(self, cfg):
        """Init for ParetoFront."""
        logging.info("start init ParetoFront")
        object_count = 2
        max_object_ids = []
        if "pareto" in cfg:
            object_count = cfg.pareto.object_count
            max_object_ids = cfg.pareto.max_object_ids
        self.sieve_columns = ['config_id', 'md5', 'config']
        for i in range(0, object_count):
            self.sieve_columns.append("score_{}".format(i))
        self.sieve_board = pd.DataFrame(columns=self.sieve_columns)
        self.max_object_ids = None
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
            nondominated = pareto.eps_sort(
                [list(pareto_board.itertuples(False))],
                objectives=self.pareto_cols,
                epsilons=None,
                maximize=self.max_object_ids)
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
        tmp_column.remove('md5')
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
        md5 = hashlib.md5(json.dumps(config_dict, sort_keys=True).encode('utf-8')).hexdigest()
        found_df = self.sieve_board[self.sieve_board['md5'].str.contains(md5)]
        if found_df.shape[0] > 0:
            return False
        else:
            save_dict = {'config_id': id, 'md5': md5, 'config': config_dict}
            self.sieve_board = self.sieve_board.append(save_dict, ignore_index=True)
            return True
