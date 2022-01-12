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

"""Defined DnetNas."""
import random
import logging
import numpy as np
from vega.core.search_algs import SearchAlgorithm
from vega.core.search_algs import ParetoFront
from vega.common import ClassFactory, ClassType
from vega.networks.model_config import ModelConfig
from vega.report import ReportServer
from .conf import DnetNasConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class DnetNas(SearchAlgorithm):
    """DnetNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    config = DnetNasConfig()

    def __init__(self, search_space=None, **kwargs):
        """Init DnetNas."""
        super(DnetNas, self).__init__(search_space, **kwargs)
        # ea or random
        self.num_mutate = self.config.policy.num_mutate
        self.random_ratio = self.config.policy.random_ratio
        self.max_sample = self.config.range.max_sample
        self.min_sample = self.config.range.min_sample
        self.sample_count = 0
        logging.info("inited DnetNas")
        self.pareto_front = ParetoFront(
            self.config.pareto.object_count, self.config.pareto.max_object_ids)
        self._best_desc_file = 'nas_model_desc.json'

        block_nas_folder = ModelConfig.models_folder.format(local_base_path=self.local_base_path)
        logging.info(f'folder: {block_nas_folder}')
        base_reports = ReportServer().load_records_from_model_folder(block_nas_folder)
        logging.info(f'base_reports: {base_reports}')
        self.base_block = base_reports[0].desc['backbone']['encoding'].split('_')[0]

    @property
    def is_completed(self):
        """Check if NAS is finished."""
        return self.sample_count > self.max_sample

    def search(self):
        """Search in search_space and return a sample."""
        sample = {}
        while sample is None or 'code' not in sample:
            pareto_dict = self.pareto_front.get_pareto_front()
            pareto_list = list(pareto_dict.values())
            if self.pareto_front.size < self.min_sample or random.random() < self.random_ratio or len(
                    pareto_list) == 0:
                sample_desc = self.search_space.sample()
                sample_desc.update({'block_coding': self.base_block})
                sample = self.codec.encode(sample_desc)
            else:
                sample = pareto_list[0]
            if sample is not None and 'code' in sample:
                code = sample['code']
                code = self.ea_sample(code)
                sample['code'] = code
            if not self.pareto_front._add_to_board(id=self.sample_count + 1,
                                                   config=sample):
                sample = None
        self.sample_count += 1
        sample_desc = self.codec.decode(sample)
        logging.info(f"sample: {sample_desc['network.backbone.encoding']}")
        return dict(worker_id=self.sample_count, encoded_desc=sample_desc)

    def ea_sample(self, code):
        """Use EA op to change a arch code.

        :param code: list of code for arch
        :type code: list
        :return: changed code
        :rtype: list
        """
        new_arch = code.copy()
        random_value = random.randint(0, 2)
        if random_value == 0:
            self._insert(new_arch)
        elif random_value == 1:
            self._remove(new_arch)
        else:
            self._swap(new_arch)

        return new_arch

    def update(self, record):
        """Use train and evaluate result to update algorithm.

        :param performance: performance value from trainer or evaluator
        """
        perf = record.get("rewards")
        worker_id = record.get("worker_id")
        logging.info("update performance={}".format(perf))
        self.pareto_front.add_pareto_score(worker_id, perf)

    def _insert(self, arch):
        """Random insert to arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        macro_coding = arch['network.backbone.macro_coding']
        macro_coding_list = list(macro_coding)
        idx = np.random.randint(low=0, high=len(macro_coding))
        macro_coding_list.insert(idx, '1')
        macro_coding_new = ''.join(macro_coding_list)
        arch['network.backbone.macro_coding'] = macro_coding_new
        print(f'insert: {macro_coding} --> {macro_coding_new}')

        return arch

    def _remove(self, arch):
        """Random remove one from arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        macro_coding = arch['network.backbone.macro_coding']
        macro_coding_list = list(macro_coding)

        while True:
            idx = np.random.randint(low=0, high=len(macro_coding))
            if macro_coding_list[idx] == '1':
                macro_coding_list.pop(idx)
                break
        macro_coding_new = ''.join(macro_coding_list)
        arch['network.backbone.macro_coding'] = macro_coding_new
        print(f'remove: {macro_coding} --> {macro_coding_new}')

        return arch

    def _swap(self, arch):
        """Random swap one in arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        macro_coding = arch['network.backbone.macro_coding']
        macro_coding_list = list(macro_coding)

        variant_indexes = []
        for index in range(len(macro_coding) - 1):
            if macro_coding[index] != macro_coding[index + 1]:
                variant_indexes.append(index)

        while True:
            idx = np.random.randint(low=0, high=len(variant_indexes))
            index = variant_indexes[idx]
            origin_code = macro_coding_list[index]
            if index == len(macro_coding) - 2 and origin_code == '-':
                continue
            else:
                break
        macro_coding_list[index] = macro_coding_list[index + 1]
        macro_coding_list[index + 1] = origin_code

        macro_coding_new = ''.join(macro_coding_list)
        arch['network.backbone.macro_coding'] = macro_coding_new
        print(f'swap: {macro_coding} --> {macro_coding_new}')

        return arch

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.max_sample
