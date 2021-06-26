# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined AshaHpo class."""

from math import log
from vega.algorithms.hpo.sha_base import ASHA
from vega.common import ClassFactory, ClassType
from vega.algorithms.hpo.hpo_base import HPOBase
from .asha_conf import AshaConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AshaHpo(HPOBase):
    """An Hpo of ASHA."""

    config = AshaConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init AshaHpo."""
        super(AshaHpo, self).__init__(search_space, **kwargs)
        num_samples = self.config.policy.num_samples
        max_epochs = self.config.policy.max_epochs
        eta = self.config.policy.eta
        if self.config.policy.total_epochs != -1:
            num_samples, max_epochs = self.design_parameter(self.config.policy.total_epochs, eta)
        self._max_samples = num_samples
        self.hpo = ASHA(self.search_space, num_samples, max_epochs, eta=eta)

    def design_parameter(self, total_epochs, eta):
        """Design parameters based on total_epochs.

        :param total_epochs: number of epochs the algorithms need.
        :type total_epochs: int, set by user.
        """
        num_samples = 1
        max_epochs = 1
        while(num_samples * (1 + log(num_samples, eta)) <= total_epochs):
            num_samples *= eta
            max_epochs *= eta
        max_epochs /= eta
        for i in range(int(num_samples / eta), num_samples + 1):
            current_budget = 0
            current_epochs = 1
            current_samples = i
            while(current_samples > 0):
                current_budget += current_samples * current_epochs
                current_samples = int(current_samples / eta)
                current_epochs *= eta
            if current_budget == total_epochs:
                num_samples = i
                break
            elif current_budget > total_epochs:
                num_samples = i - 1
                break
        return num_samples, max_epochs

    @property
    def max_samples(self):
        """Get max samples number."""
        return self._max_samples
