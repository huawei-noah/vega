# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""
BOSS: Bayesian Optimization via Sub-Sampling for Hyperparameter Optimization.

    code-block:: python
    Detail of BOSS:
    Input: maximum budget R; ratio eta.
    Output: the configuration with the best performance.
    1.Initialize the surrogate model and the acquisition function with a uniform distribution.
    2.s_max = math.floor(log R / log eta), B = (s_max + 1) * R
    3.for s = s_max, s_max - 1, ..., 0 hyperparameter_space
    4.    K = math.ceil(B * pow(eta, s) / (R * (s + 1))), b = R * pow(eta, -s)
    5.    Sampling K configurations C from the acquisition function.
    6.    Call SS with (C, b, eta)
    7.    Use the output data from SS to refit the model and update the acquisition function.
    8.end for
"""

import operator
import math
from .sha_base import ShaBase
import random
from .ssa import SSA
from .tuner import TunerBuilder
from vega.core.hyperparameter_space import HyperparameterSpace


class BOSS(ShaBase):
    """BOSS: Bayesian Optimization and SubSampling, combines Bayesian optimization and SubSampling.

    :param hyperparameter_space: a pre-defined search space.
    :type hyperparameter_space: object, instance os `HyperparameterSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param int repeat_times: repeat times of total iters.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, hyperparameter_space, config_count, max_epochs, repeat_times, min_epochs=1,
                 eta=3):
        """Init BOSS."""
        super().__init__(hyperparameter_space, config_count, max_epochs, min_epochs,
                         eta)
        # init all the configs
        self.repeat_times = repeat_times
        self.hp = TunerBuilder(hyperparameter_space, tuner='GPEI')
        self.iter_list, self.min_epoch_list = self._get_total_iters(
            config_count, max_epochs, self.repeat_times, min_epochs, eta)
        self.config_dict = {}

        # init the empty ssa config list, all ssa object need to be set_config_list
        self.ssa_list = self._get_ssa_list(self.iter_list, self.min_epoch_list, self.repeat_times)
        # init the first ssa with first config list
        self.config_dict[0] = self.get_hyperparameter_space(self.iter_list[0])
        self.ssa_list[0].set_config_list(self.config_dict[0], start_id=0)
        return

    def _get_total_iters(self, config_count, max_epochs, repeat_times, min_epochs=1, eta=3):
        """Calculate each rung for all iters of SubSampling algorithm.

        :param config_count: int, Total config count to optimize.
        :param max_epochs: int, max epochs of evaluate function.
        :param repeat_times: int, repeat times of total iters.
        :param min_epochs: int, the epoch start with min epochs, default 1.
        :param eta: int, default 3.
        :return:  it_list, ep_list
        """
        each_count = (config_count + repeat_times - 1) // repeat_times
        rest_count = config_count
        count_list = []
        for i in range(repeat_times):
            if rest_count >= each_count:
                count_list.append(each_count)
                rest_count -= each_count
            else:
                count_list.append(rest_count)
        iter_list_hl = []
        min_ep_list_hl = []
        for i in range(repeat_times):
            diff = 1
            iter = -1
            iter_list = []
            min_ep_list = []
            while diff > 0:
                iter = iter + 1
                diff = count_list[i] - (math.pow(eta, iter + 1) - 1) / (eta - 1)
                if diff > 0:
                    iter_list.append(int(math.pow(eta, iter)))
                else:
                    if len(iter_list) == 0:
                        iter_list.append(int(count_list[i]))
                    else:
                        iter_list.append(int(
                            count_list[i] - (math.pow(eta, iter) - 1) / (eta - 1)))
            iter_list.sort(reverse=True)
            for i in range(len(iter_list)):
                temp_ep = int(min_epochs * math.pow(eta, i))
                if temp_ep > max_epochs:
                    temp_ep = max_epochs
                min_ep_list.append(temp_ep)
            iter_list_hl.append(iter_list)
            min_ep_list_hl.append(min_ep_list)
        it_list = []
        ep_list = []
        for i in range(repeat_times):
            for j in range(len(iter_list_hl[i])):
                it_list.append(iter_list_hl[i][j])
                ep_list.append(min_ep_list_hl[i][j])
        return it_list, ep_list

    def _get_ssa_list(self, iter_list, min_epoch_list, repeat_times):
        """Init a list contain different SSA object for different iter.

        each have a part of configs from total config_list.

        :param iter_list: iter list for function _get_total_iters
        :param min_epoch_list: result of function _get_total_iters
        :return: list[SSA]
            a list of SSA objects for different iters.
        """
        ssa_list = []
        for i in range(len(iter_list)):
            tmp_ds = HyperparameterSpace()
            tmp_ssa = SSA(tmp_ds, iter_list[i], self.max_epochs,
                          min_epoch_list[i], self.eta, empty=True)
            ssa_list.append(tmp_ssa)
        return ssa_list

    def _set_next_ssa(self):
        """Use hp propose next iter configs list.

        and reset the next iter's ssa oject.
        """
        if self.is_completed or self.current_iter >= len(self.iter_list):
            return
        # add new iter
        self.current_iter = self.current_iter + 1
        iter = self.current_iter
        # update the hp model with previous iter scores
        self._set_hp(iter)
        # use hp algorithm porpose next iter's config list.
        configs = self.hp.propose(self.iter_list[iter])
        self.config_dict[iter] = configs
        count = 0
        for i in range(0, iter):
            count += self.iter_list[i]
        self.ssa_list[iter].set_config_list(self.config_dict[iter],
                                            start_id=count)
        return

    def _set_hp(self, iter):
        """Use iter ssa results to train a new hp model."""
        next_budget = self.min_epoch_list[iter]
        next_config = self.iter_list[iter]
        cn = int(math.sqrt(math.log(next_config * self.eta / (self.eta - 1))))
        if cn != 1:
            for i in range(cn - 1):
                next_budget *= self.eta
        for j in range(iter):
            ssa = self.ssa_list[j]
            rung_list = []
            for i in range(len(ssa.budget_list)):
                if ssa.budget_list[i] >= next_budget and ssa.budget_list[i] / 3. < next_budget:
                    rung_list.append(i)
            for i in ssa.all_config_dict:
                x = ssa.all_config_dict[i]
                score_df = ssa.sieve_board[ssa.sieve_board['config_id'] == i]
                for k in rung_list:
                    score_df = score_df[score_df['rung_id'] == k]
                    if score_df.empty:
                        continue
                    else:
                        y = float(score_df['score'])
                    self.hp.add(x, y)
        return self.hp

    def best_config(self):
        """Get current best score config.

        :return:  dict
            {'config_id': int,
                'score': float,
                'configs': dict}
            config_id, score, and configs of the current best config.
        """
        if self.total_propose == 0:
            idx = random.randint(0, len(self.hyperparameter_list))
            result = {'config_id': idx,
                      'score': -1 * float('inf'),
                      'configs': self.hyperparameter_list[idx]}
            return result
        else:
            best_iter_id = max(self.best_score_dict.items(),
                               key=operator.itemgetter(1))[0]
            return self.ssa_list[best_iter_id].best_config()

    def add_score(self, config_id, rung_id, score):
        """Add score into best score dict and board of ssa bracket.

        :param config_id: config id in broad data frame
        :param rung_id: rung id in broad data frame
        :param score: the best score need to set
        """
        iter_id = self.current_iter
        self.ssa_list[iter_id].add_score(config_id, rung_id, score)
        # check if current iter is completed
        if self._check_completed(iter_id):
            if iter_id == len(self.iter_list) - 1:
                self.is_completed = True
            else:
                self._set_next_ssa()
        # for best config
        if iter_id not in self.best_score_dict:
            self.best_score_dict[iter_id] = score
        elif score > self.best_score_dict[iter_id]:
            self.best_score_dict[iter_id] = score
        return

    def propose(self):
        """Propose the next hyper parameter for ssa bracket.

        :return: list
        """
        iter_id = self.current_iter
        if self._check_completed(iter_id):
            return None
        results = self.ssa_list[iter_id].propose()
        if results is not None:
            self.total_propose = self.total_propose + 1
        return results

    def _check_completed(self, iter_id):
        """Check all ssa task completed.

        :param iter_id: the iter id of ssa bracket
        :return: True/False
        """
        if iter_id != self.current_iter:
            raise ValueError("iter_id not equal to current iter id!")
        return self.ssa_list[iter_id].is_completed
