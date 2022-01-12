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

import math
import logging
from vega.common.class_factory import ClassFactory, ClassType
from .asha import ASHA
from .tuner import TunerBuilder
from ..ea.ga import GeneticAlgorithm
from .sha_base import ShaBase
from .status_type import StatusType


logger = logging.getLogger(__name__)


class BOHB(ShaBase):
    """BOHB: Bayesian Optimization and Hyperband, combines Bayesian optimization and Hyperband.

    :param search_space: a pre-defined search space.
    :type search_space: object, instance os `SearchSpace`.
    :param int num_samples: Total config or hyperparameter count.
    :param int max_epochs: `max_epochs` is the max epoch that hpo provide.
    :param min_epochs: `min_epochs` is the init min epoch.
    :type min_epochs: int, default is 1.
    :param eta: rung base `eta`.
    :type eta: int, default is 3.
    """

    def __init__(self, search_space, num_samples, max_epochs, repeat_times,
                 min_epochs=1, eta=3, multi_obj=False, random_samples=None,
                 prob_crossover=0.6, prob_mutatation=0.2, tuner="RF"):
        """Init BOHB."""
        super().__init__(search_space, num_samples, max_epochs, min_epochs, eta)
        # init all the configs
        self.repeat_times = repeat_times
        self.max_epochs = max_epochs
        self.iter_list, self.min_epoch_list = self._get_total_iters(
            num_samples, max_epochs, self.repeat_times, min_epochs, eta)
        self.additional_samples = self._get_additional_samples(eta)
        if random_samples is not None:
            self.random_samples = random_samples
        else:
            self.random_samples = max(self.iter_list[0][0], 2)
        self.tuner_name = "GA" if multi_obj else tuner
        logger.info(f"bohb info, iter list: {self.iter_list}, min epoch list: {self.min_epoch_list}, "
                    f"addition samples: {self.additional_samples}, tuner: {self.tuner_name}, "
                    f"random samples: {self.random_samples}")
        # create sha list
        self.sha_list = self._create_sha_list(
            search_space, self.iter_list, self.min_epoch_list, self.repeat_times)
        # create tuner
        if multi_obj:
            self.tuner = GeneticAlgorithm(
                search_space, random_samples=self.random_samples,
                prob_crossover=prob_crossover, prob_mutatation=prob_mutatation)
        elif self.tuner_name == "hebo":
            self.tuner = ClassFactory.get_cls(ClassType.SEARCH_ALGORITHM, "HeboAdaptor")(search_space)
        else:
            self.tuner = TunerBuilder(search_space, tuner=tuner)

    def _get_total_iters(self, num_samples, max_epochs, repeat_times, min_epochs=1, eta=3):
        """Calculate each rung for all iters of Hyper Band algorithm.

        n = |(B/R)*η^s/(s+1)e|,  r = R*η^(−s)

        :param num_samples: int, Total config count to optimize.
        :param max_epochs: int, max epochs of evaluate function.
        :param min_epochs: int, the epoch start with min epochs, default 1.
        :param eta: int, default 3.
        :return:  iter_list, min_ep_list
        """
        each_count = (num_samples + repeat_times - 1) // repeat_times
        rest_count = num_samples
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
            for j in range(len(iter_list)):
                temp_ep = int(min_epochs * math.pow(eta, j))
                if temp_ep > max_epochs:
                    temp_ep = max_epochs
                min_ep_list.append(temp_ep)
            iter_list_hl.append(iter_list)
            min_ep_list_hl.append(min_ep_list)
        return iter_list_hl, min_ep_list_hl

    def _get_additional_samples(self, eta):
        additional_samples = []
        for origin_list in self.iter_list:
            additional_list = [0 for _ in range(len(origin_list))]
            origin_value = origin_list[0]
            for i in range(1, len(origin_list)):
                value = math.ceil(origin_value / 2 / i)
                sub_list = [value]
                while value > 1:
                    value = math.floor(value / eta)
                    if value > 0:
                        sub_list.append(value)
                    else:
                        break
                for j, value in enumerate(sub_list):
                    if j + i < len(additional_list):
                        additional_list[j + i] += value
            additional_samples.append(additional_list)
        return additional_samples

    def _create_sha_list(self, search_space, iter_list, min_epoch_list, repeat_times):
        """Init a list contain different SHA object for different iter.

        each have a part of configs from total config_list.

        :param iter_list: iter list for function _get_total_iters
        :param min_epoch_list: result of function _get_total_iters
        :return: list[SHA]
            a list of SHA objects for different iters.
        """
        sha_list = []
        for i in range(repeat_times):
            tmp_sha = ASHA(
                search_space,
                iter_list[i][0],
                int(min_epoch_list[i][0] * math.pow(self.eta, len(iter_list[i]) - 1)),
                min_epoch_list[i][0],
                self.eta,
                random_config=False,
                start_config_id=i * (iter_list[i][0] + sum(self.additional_samples[i])))
            tmp_sha.get_config = self.get_config
            sha_list.append(tmp_sha)
        return sha_list

    def get_config(self, config_id):
        """Get config."""
        if config_id in self.config_dict:
            return self.config_dict[config_id]
        else:
            board = self.sha_list[0].sieve_board
            df = board.loc[board["status"].isin([StatusType.FINISHED, StatusType.PORMOTED])]
            finished_models = df.shape[0]
            if finished_models < self.random_samples and self.tuner_name != "hebo":
                config = self.get_hyperparameters(1)[0]
                logger.info("random sample")
            else:
                config = self.tuner.propose(1)[0]
                logger.info("generate sample from model")
            self.config_dict[config_id] = config
            self.sha_list[self.current_iter].all_config_dict = self.config_dict
            return config

    def _update_hp(self, config, score):
        """Use iter sha results to train a new hp model."""
        self.tuner.add(config, score)

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
                self.current_iter += 1

        config = self.get_config(config_id)
        self._update_hp(config, score)

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

        for rung_id in range(len(self.additional_samples[iter_id])):
            if self.additional_samples[iter_id][rung_id] != 0:
                self.additional_samples[iter_id][rung_id] -= 1
                # new config
                config_id = len(self.config_dict)
                config = self.get_config(config_id)
                self.config_dict[config_id] = config
                self.sha_list[self.current_iter].all_config_dict = self.config_dict
                results = {
                    'config_id': config_id,
                    'rung_id': rung_id,
                    'configs': config,
                    'epoch': self.min_epoch_list[iter_id][rung_id],
                }
                self.sha_list[iter_id]._change_status(
                    rung_id=rung_id, config_id=config_id, status=StatusType.RUNNING)
                self.sha_list[iter_id].total_propose += 1
                self.total_propose += 1
                logger.info(f"propose new config, tuner: {self.tuner_name}\n{self.sha_list[iter_id].sieve_board}")
                return results

        return None

    def next_rung(self, config_id):
        """Query next rung config."""
        iter_id = self.current_iter
        if self._check_completed(iter_id):
            return None
        results = self.sha_list[iter_id].next_rung(config_id)
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
        return self.sha_list[iter_id].is_completed and self.additional_samples[iter_id][-1] == 0
