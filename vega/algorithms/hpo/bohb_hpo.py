# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined BohbHpo class."""
from math import pow
from vega.algorithms.hpo.common import BOHB
from zeus.common import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .bohb_conf import BohbConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BohbHpo(HPOBase):
    """An Hpo of BOHB, inherit from HpoGenerator."""

    config = BohbConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init BohbHpo."""
        super(BohbHpo, self).__init__(search_space, **kwargs)
        num_samples = self.config.policy.num_samples
        max_epochs = self.config.policy.max_epochs
        if self.config.policy.total_epochs != -1:
            num_samples, max_epochs = self.design_parameter()
        self.hpo = BOHB(self.search_space,
                        num_samples,
                        max_epochs,
                        self.config.policy.repeat_times,
                        self.config.policy.min_epochs,
                        self.config.policy.eta)

    def design_parameter(self):
        """Design parameters based on total_epochs.

        :param total_epochs: number of epochs the algorithms need.
        :type total_epochs: int, set by user.
        """
        total_epochs = self.config.policy.total_epochs
        num_samples = 1
        iter_list = []
        min_epoch_list = []
        for num_samples in range(4, total_epochs):
            iter_list, min_epoch_list = self.get_iter_epoch_list(
                num_samples)
            current_budget = 0
            for i in range(len(iter_list)):
                current_samples = iter_list[i]
                current_epochs = min_epoch_list[i]
                while(current_samples > 0):
                    current_budget += current_samples * current_epochs
                    current_samples = int(current_samples / 3)
                    current_epochs *= 3
            if current_budget == total_epochs:
                break
            elif current_budget > total_epochs:
                num_samples -= 1
                break
        max_epochs = max(min_epoch_list)
        return num_samples, max_epochs

    def get_iter_epoch_list(self, num_samples):
        """Calculate each rung for all iters of Hyper Band algorithm.

        :param num_samples: int, Total config count to optimize.
        :param repeat_times: int, repeat times of algorithm.
        :return:  iter_list, min_ep_list
        """
        min_epochs = self.config.policy.min_epochs
        eta = self.config.policy.eta
        repeat_times = self.config.policy.repeat_times
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
            for i in range(len(iter_list)):
                temp_ep = int(min_epochs * pow(eta, i))
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
