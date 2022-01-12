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
Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.

https://arxiv.org/pdf/1603.06560.pdf
.. code-block:: python
    Detail of Hyperband:

    Input: R, η (default η = 3)
    initialization : s_max = |log_η(R)|,  B = (s_max + 1)R

    for s ∈ {smax, smax − 1, . . . , 0} do
        n = |(B/R)*η^s/(s+1)e|,  r = R*η^(−s)
        // begin SuccessiveHalving with (n, r) inner loop
        T =get_hyperparameter_configuration(n)
        for i ∈ {0, . . . , s} do
            n_i = |n*η^(−i)|
            r_i = r*η^(i)
            L = {run_then_return_val_loss(t, r_i) : t ∈ T}
            T =top_k(T, L, |n_i/η|)
        end
    end
    return Configuration with the smallest intermediate loss seen so far
"""
import operator
import math
import random
from vega.core.search_space import SearchSpace
from .sha_base import ShaBase
from .sha import SHA


class HyperBand(ShaBase):
    """Current HyperBand need to rely on SHA algorithm to process inner loop.

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int config_count: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, hyperparameter_space, config_count, max_epochs, min_epochs=1,
                 eta=3):
        """Init for HyperBand."""
        super().__init__(hyperparameter_space, config_count, max_epochs, min_epochs,
                         eta)
        self.iter_list, self.min_epoch_list = self._get_total_iters(
            config_count, max_epochs, min_epochs, eta)
        self.sha_list = self._get_sha_list(self.iter_list,
                                           self.hyperparameter_list,
                                           self.min_epoch_list)
        return

    def _get_total_iters(self, config_count, max_epochs, min_epochs=1, eta=3):
        """Calculate each rung for all iters of Hyper Band algorithm.

          n = |(B/R)*η^s/(s+1)e|,  r = R*η^(−s)

        :param config_count: int, Total config count to optimize.
        :param max_epochs: int, max epochs of evaluate function
        :param min_epochs: int, the epoch start with min epochs, default 1
        :param eta:
        :return:  iter_list, min_ep_list
        """
        diff = 1
        iter = -1
        iter_list = []
        min_ep_list = []
        while diff > 0:
            iter = iter + 1
            diff = config_count - (math.pow(eta, iter + 1) - 1) / (eta - 1)
            if diff > 0:
                iter_list.append(int(math.pow(eta, iter)))
            else:
                if len(iter_list) == 0:
                    iter_list.append(int(config_count))
                else:
                    iter_list[-1] += int(
                        config_count - (math.pow(eta, iter) - 1) / (eta - 1))
        iter_list.reverse()
        for i in range(len(iter_list)):
            temp_ep = int(min_epochs * math.pow(eta, i))
            if temp_ep > max_epochs:
                temp_ep = max_epochs
            min_ep_list.append(temp_ep)
        return iter_list, min_ep_list

    def _get_sha_list(self, iter_list, config_list, min_epoch_list):
        """Split the total config_list according to previous iter_list.

        and init a list contain different SHA object for different iter,
        each have a part of configs from total config_list.

        :param iter_list: result of _get_total_iters function
        :param config_list: all configs
        :param min_epoch_list: result from _get_total_iters function
        :return: list[SHA]
            a list of SHA objects for different iters.
        """
        config_dict = {}
        count = 0
        sha_list = []
        for i in range(len(iter_list)):
            config_dict[i] = config_list[count:count + iter_list[i]]
            #
            tmp_hps = SearchSpace()
            tmp_sha = SHA(tmp_hps, iter_list[i], self.max_epochs,
                          min_epoch_list[i], self.eta, empty=True)
            # init the sha config list
            tmp_sha.set_config_list(config_dict[i], start_id=count)
            sha_list.append(tmp_sha)
            count += iter_list[i]
        return sha_list

    def best_config(self):
        """Get current best score config.

        :return: dict
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
        :return:
        """
        iter_id = self.current_iter
        self.sha_list[iter_id].add_score(config_id, rung_id, score)
        # check if current iter is completed
        if self._check_completed(iter_id):
            if iter_id == len(self.iter_list) - 1:
                self.is_completed = True
            else:
                self.current_iter = self.current_iter + 1
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
