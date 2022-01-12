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

"""The second stage of SP-NAS."""

import copy
import logging
from random import SystemRandom
import numpy as np
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from vega.report import ReportServer
from .conf import SpNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class SpNasS(SearchAlgorithm):
    """First Stage of SPNAS."""

    config = SpNasConfig()

    def __init__(self, search_space=None):
        super(SpNasS, self).__init__(search_space)
        self.sample_count = 0
        self.max_sample = self.config.max_sample

    @property
    def is_completed(self):
        """Check sampling if finished."""
        return self.sample_count >= self.max_sample

    def search(self):
        """Search a sample."""
        pareto_records = ReportServer().get_pareto_front_records(choice='normal')
        best_record = pareto_records[0] if pareto_records else None
        desc = self.search_space.sample()
        arch_code = desc.get('network.backbone.code')
        if best_record:
            desc['network.backbone.weight_file'] = best_record.weights_file
        desc['network.backbone.code'] = self._mutate_serialnet(arch_code)
        self.sample_count += 1
        logging.info("desc:{}".format(desc))
        return dict(worker_id=self.sample_count, encoded_desc=desc)

    @property
    def max_samples(self):
        """Return the max number of samples."""
        return self.max_sample

    def _mutate_serialnet(self, arch):
        """Swap & Expend operation in Serial-level searching."""

        def is_valid(arch):
            stages = arch.split('-')
            for stage in stages:
                if len(stage) == 0:
                    return False
            return True

        def expend(arc):
            idx = np.random.randint(low=1, high=len(arc))
            arc = arc[:idx] + '1' + arc[idx:]
            return arc, idx

        def swap(arc, len_step=3):
            is_not_valid = True
            arc_origin = copy.deepcopy(arc)
            temp = arc.split('-')
            num_insert = len(temp) - 1
            while is_not_valid or arc == arc_origin:
                next_start = 0
                arc = list(''.join(temp))
                for i in range(num_insert):
                    pos = arc_origin[next_start:].find('-') + next_start
                    if arc_origin[pos] == '-':
                        max_step = min(len_step, max(len(temp[i]), len(temp[i + 1])))
                        step_range = list(range(-1 * max_step, max_step))
                        step = SystemRandom().choice(step_range)
                        next_start = pos + 1
                        pos = pos + step
                        arc.insert(pos, '-')
                    else:
                        raise ValueError("Wrong '-' is found!")
                arc = ''.join(arc)
                is_not_valid = (not is_valid(arc))
            return arc

        arch_origin = arch
        success = False
        k = 0
        while not success:
            k += 1
            arch = arch_origin
            ops = []
            for i in range(self.config.num_mutate):
                op_idx = np.random.randint(low=0, high=2)
                adds_thresh_ = self.config.add_stage_ratio if len(arch.split('-')) < self.config.max_stages else 1
                if op_idx == 0 and SystemRandom().random() > self.config.expend_ratio:
                    arch, idx = expend(arch)
                    arch, idx = expend(arch)
                    ops.append('expend')
                elif op_idx == 1:
                    arch = swap(arch)
                    ops.append('swap')
                elif op_idx == 2 and SystemRandom().random() > adds_thresh_:
                    arch = arch + '-1'
                    ops.append('add stage')
                else:
                    ops.append('Do Nothing')
            success = arch != arch_origin
            flag = 'Success' if success else 'Failed'
            logging.info('Serial-level Sample{}: {}. {}.'.format(k, ' -> '.join(ops), flag))
        return arch
