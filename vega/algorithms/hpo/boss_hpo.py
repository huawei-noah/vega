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

"""Defined BohbHpo class."""
from math import log, pow, sqrt
from vega.algorithms.hpo.sha_base import BOSS
from vega.common import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .boss_conf import BossConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BossHpo(HPOBase):
    """An Hpo of BOSS."""

    config = BossConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BossHpo."""
        super(BossHpo, self).__init__(search_space, **kwargs)
        if self.config.policy.max_epochs < 3:
            raise ValueError('set total_epochs illegal, count not less than 3!')
        num_samples = self.config.policy.num_samples
        max_epochs = self.config.policy.max_epochs
        repeat_times = self.config.policy.repeat_times
        if self.config.policy.total_epochs != -1:
            total_epochs = self.config.policy.total_epochs
            num_samples, max_epochs = self.design_parameter(total_epochs, repeat_times)
        self._max_samples = num_samples
        self.hpo = BOSS(self.search_space, num_samples, max_epochs, repeat_times, tuner=self.config.tuner)

    def design_parameter(self, total_epochs, repeat_times):
        """Design parameters based on total_epochs.

        :param total_epochs: number of epochs the algorithms need.
        :type total_epochs: int, set by user.
        """
        eta = 3
        num_samples = 1
        iter_list = []
        min_epoch_list = []
        for num_samples in range(4, total_epochs):
            iter_list, min_epoch_list = self.get_iter_epoch_list(
                num_samples, repeat_times)
            current_budget = 0
            for i in range(len(iter_list)):
                current_samples = iter_list[i]
                current_epochs = min_epoch_list[i]
                cn = int(sqrt(log(current_samples * 3 / 2)))
                min_epochs = current_epochs
                if cn != 1:
                    for _ in range(cn - 1):
                        min_epochs *= eta
                while(current_samples > 0):
                    valid_epochs = max(min_epochs, current_epochs)
                    current_budget += current_samples * valid_epochs
                    current_samples = int(current_samples / 3)
                    current_epochs *= 3
            if current_budget == total_epochs:
                break
            elif current_budget > total_epochs:
                num_samples -= 1
                break
        max_epochs = max(min_epoch_list)
        return num_samples, max_epochs

    def get_iter_epoch_list(self, num_samples, repeat_times):
        """Calculate each rung for all iters of Hyper Band algorithm.

        :param num_samples: int, Total config count to optimize.
        :param repeat_times: int, repeat times of algorithm.
        :return:  iter_list, min_ep_list
        """
        min_epochs = 1
        eta = 3
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
                diff = count_list[i] - (pow(eta, iter + 1) - 1) / (eta - 1)
                if diff > 0:
                    iter_list.append(int(pow(eta, iter)))
                else:
                    if len(iter_list) == 0:
                        iter_list.append(int(count_list[i]))
                    else:
                        iter_list.append(int(
                            count_list[i] - (pow(eta, iter) - 1) / (eta - 1)))
            iter_list.sort(reverse=True)
            for j in range(len(iter_list)):
                temp_ep = int(min_epochs * pow(eta, j))
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

    @property
    def max_samples(self):
        """Get max samples number."""
        return self._max_samples
