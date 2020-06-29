# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.


"""
BOHB: Robust and Efficient Hyperparameter Optimization at Scale.

https://ml.informatik.uni-freiburg.de/papers/18-ICML-BOHB.pdf
https://www.automl.org/automl/bohb/
.. code-block:: python
    Detail of BOHB:
    Input: observations D, fraction of random runs ρ,
            percentile q, number of samples N_s,
            minimum number of points N_min to build a model,
            and bandwidth factor b_w
    Output: next configuration to evaluate
    if rand() < ρ:
        then return random configuration
    b = arg_max{D_b : |D_b| ≥ N_min + 2}
    if b = ∅:
        then return random configuration
    fit KDEs according to Eqs. (2) and (3)
    draw N_s samples according to l'(x) (see text)
    return sample with highest ratio l(x)/g(x)
    Eq(2):
        l(x) = p(y < α|x, D)
        g(x) = p(y > α|x, D)
    Eq(3):
        N_(b,l) = max(N_min, q · N_b)
        N_(b,g) = max(N_min, N_b − N_(b,l))
"""

import operator
import math
import numpy as np
from .sha_base import ShaBase
import random
from .sha import SHA
from .tuner import TunerBuilder
from vega.core.hyperparameter_space import HyperparameterSpace


class BOHB(ShaBase):
    """BOHB: Bayesian Optimization and Hyperband, combines Bayesian optimization and Hyperband.

    :param hyperparameter_space: a pre-defined search space.
    :type hyperparameter_space: object, instance os `HyperparameterSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, hyperparameter_space, config_count, max_epochs, repeat_times, min_epochs=1,
                 eta=3):
        """Init BOHB."""
        super().__init__(hyperparameter_space, config_count, max_epochs, min_epochs,
                         eta)
        # init all the configs
        self.repeat_times = repeat_times
        self.hp = TunerBuilder(hyperparameter_space, tuner='GPEI')
        self.iter_list, self.min_epoch_list = self._get_total_iters(
            config_count, max_epochs, self.repeat_times, min_epochs, eta)
        self.config_dict = {}

        # init the empty sha config list, all sha object need to be set_config_list
        self.sha_list = self._get_sha_list(self.iter_list, self.min_epoch_list, self.repeat_times)
        # init the first sha with first config list
        self.config_dict[0] = self.get_hyperparameter_space(self.iter_list[0])
        self.sha_list[0].set_config_list(self.config_dict[0], start_id=0)
        return

    def _get_total_iters(self, config_count, max_epochs, repeat_times, min_epochs=1, eta=3):
        """Calculate each rung for all iters of Hyper Band algorithm.

        n = |(B/R)*η^s/(s+1)e|,  r = R*η^(−s)

        :param config_count: int, Total config count to optimize.
        :param max_epochs: int, max epochs of evaluate function.
        :param min_epochs: int, the epoch start with min epochs, default 1.
        :param eta: int, default 3.
        :return:  iter_list, min_ep_list
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

    def _get_sha_list(self, iter_list, min_epoch_list, repeat_times):
        """Init a list contain different SHA object for different iter.

        each have a part of configs from total config_list.

        :param iter_list: iter list for function _get_total_iters
        :param min_epoch_list: result of function _get_total_iters
        :return: list[SHA]
            a list of SHA objects for different iters.
        """
        sha_list = []
        for i in range(len(iter_list)):
            tmp_hps = HyperparameterSpace()
            tmp_sha = SHA(tmp_hps, iter_list[i], self.max_epochs,
                          min_epoch_list[i], self.eta, empty=True)
            sha_list.append(tmp_sha)
        return sha_list

    def _set_next_sha(self):
        """Use hp propose next iter configs list, and reset the next iter's sha oject."""
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
        self.sha_list[iter].set_config_list(self.config_dict[iter],
                                            start_id=count)
        return

    def _set_hp(self, iter):
        """Use iter sha results to train a new hp model."""
        next_budget = self.min_epoch_list[iter]
        for j in range(iter):
            sha = self.sha_list[j]
            current_budget = self.min_epoch_list[j]
            current_config_num = self.iter_list[j]
            current_rung_id = -1
            while (current_budget <= next_budget) and (current_config_num >= 1):
                current_budget *= self.eta
                current_config_num //= self.eta
                current_rung_id += 1
            if current_rung_id == -1:
                continue
            for i in sha.all_config_dict:
                x = sha.all_config_dict[i]
                score_df = sha.sieve_board[sha.sieve_board['config_id'] == i]
                score_df = score_df[score_df['rung_id'] == current_rung_id]
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
            return self.sha_list[best_iter_id].best_config()

    def add_score(self, config_id, rung_id, score):
        """Add score into best score dict and board of sha bracket.

        :param config_id: config id in broad data frame
        :param rung_id: rung id in broad data frame
        :param score: the best score need to set
        """
        iter_id = self.current_iter
        self.sha_list[iter_id].add_score(config_id, rung_id, score)
        # check if current iter is completed
        if self._check_completed(iter_id):
            if iter_id == len(self.iter_list) - 1:
                self.is_completed = True
            else:
                self._set_next_sha()
        # for best config
        if iter_id not in self.best_score_dict:
            self.best_score_dict[iter_id] = score
        elif score > self.best_score_dict[iter_id]:
            self.best_score_dict[iter_id] = score
        return

    def propose(self):
        """Propose the next hyper parameter for sha bracket.

        :return: list
        """
        iter_id = self.current_iter
        if self._check_completed(iter_id):
            return None
        results = self.sha_list[iter_id].propose()
        if results is not None:
            self.total_propose = self.total_propose + 1
        return results

    def _check_completed(self, iter_id):
        """Check all sha task completed.

        :param iter_id: the iter id of sha bracket
        :return: True/False
        """
        if iter_id != self.current_iter:
            raise ValueError("iter_id not equal to current iter id!")
        return self.sha_list[iter_id].is_completed
