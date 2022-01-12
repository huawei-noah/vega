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

"""Remaining Runtime Estimator."""
import time
import logging
from vega.common import Config


class RuntimeEstimator(object):
    """Runtime Estimator.

    :param types: runtime types, default ['epoch', 'train']
    :type types: list or tuple
    :param max_steps: max steps of each type
    :type max_steps: list or tuple
    """

    def __init__(self, types=None, max_steps=None):
        if types is None:
            types = ['epoch', 'train']
        if max_steps is None:
            max_steps = [0, 0]
        self.estimator = Config()
        if not isinstance(types, list) or not isinstance(max_steps, list):
            types = [types]
            max_steps = [max_steps]
        if len(types) != len(max_steps):
            raise Exception('types length must equal to max_step')
        for type, max_step in zip(types, max_steps):
            self.add_runtime_est(type, max_step)

    def add_runtime_est(self, type, max_step):
        """Add new type of runtime estimator.

        :param type: runtime type
        :type type: str
        :param max_step: max step of new type
        :type type: int
        """
        if type in self.estimator:
            logging.warning('type %s has already in estimator', type)
            return
        self.estimator[type] = Config()
        self.estimator[type].start_time = None
        self.estimator[type].current_time = None
        self.estimator[type].start_step = 0
        self.estimator[type].current_step = 0
        self.estimator[type].max_step = max_step

    def mark_start_time(self, type, step):
        """Mark the start time.

        :param type: runtime type
        :type type: str
        :param step: start step
        :type step: int
        """
        if type not in self.estimator:
            logging.error('runtime estimator has no type %s', type)
            return
        self.estimator[type].start_time = time.time()
        self.estimator[type].start_step = step

    def remaining_time(self, type, step):
        """Calculate the remaining run time.

        :param type: runtime type
        :type type: str
        :param step: start step
        :type step: int
        :return: remaining time
        :rtype: datetime.timedelta
        """
        if type not in self.estimator:
            logging.error('runtime estimator has no type %s', type)
            return
        run_est = self.estimator[type]
        if run_est.start_time is None:
            logging.error('runtime estimator start time has not been marked')
            return
        if step == run_est.start_step:
            logging.warning('current step should not equal to start step')
            return
        run_est.current_time = time.time()
        interval = run_est.current_time - run_est.start_time
        run_est.current_step = step
        run_steps = run_est.current_step - run_est.start_step
        remain_time = interval * (run_est.max_step - run_est.current_step) / run_steps
        return remain_time / 3600

    def using_time(self, type):
        """Get using time."""
        return (self.estimator[type].current_time - self.estimator[type].start_time) / 3600
