# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined BackboneNas."""
import os
import random
import numpy as np
import logging
import json
import copy
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.search_algs.ea import EvolutionAlgorithm
from vega.search_space.search_algs.random_search import RandomSearchAlgorithm
from vega.search_space.search_algs.pareto_front import ParetoFront
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class BackboneNas(SearchAlgorithm):
    """BackboneNas.

    :param search_space: input search_space
    :type: SeachSpace
    """

    def __init__(self, search_space=None):
        """Init BackboneNas."""
        super(BackboneNas, self).__init__(search_space)
        # ea or random
        self.search_space = search_space
        self.codec = Codec(self.cfg.codec, search_space)
        self.num_mutate = self.policy.num_mutate
        self.random_ratio = self.policy.random_ratio
        self.max_sample = self.range.max_sample
        self.min_sample = self.range.min_sample
        self.sample_count = 0
        logging.info("inited BackboneNas")
        self.pareto_front = ParetoFront(self.cfg)
        self.random_search = RandomSearchAlgorithm(self.search_space)
        self._best_desc_file = 'nas_model_desc.json'
        if 'best_desc_file' in self.cfg and self.cfg.best_desc_file is not None:
            self._best_desc_file = self.cfg.best_desc_file

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
                sample_desc = self.random_search.search()
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
        logging.info(sample)
        sample_desc = self.codec.decode(sample)
        return self.sample_count, NetworkDesc(sample_desc)

    def random_sample(self):
        """Random sample from search_space."""
        sample_desc = self.random_search.search()
        sample = self.codec.encode(sample_desc, is_random=True)
        return sample

    def ea_sample(self, code):
        """Use EA op to change a arch code.

        :param code: list of code for arch
        :type code: list
        :return: changed code
        :rtype: list
        """
        new_arch = code.copy()
        self._insert(new_arch)
        self._remove(new_arch)
        self._swap(new_arch[0], self.num_mutate // 2)
        self._swap(new_arch[1], self.num_mutate // 2)
        return new_arch

    def update(self, worker_result_path):
        """Use train and evaluate result to update algorithm.

        :param worker_result_path: current result path
        :type: str
        """
        step_name = os.path.basename(os.path.dirname(worker_result_path))
        config_id = int(os.path.basename(worker_result_path))
        performance = self._get_performance(step_name, config_id)
        logging.info("update performance={}".format(performance))
        self.pareto_front.add_pareto_score(config_id, performance)
        self.save_output(self.local_output_path)
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_base_path, self.backup_base_path)

    def _get_performance(self, step_name, worker_id):
        saved_folder = self.get_local_worker_path(step_name, worker_id)
        performance_file = FileOps.join_path(saved_folder, "performance.txt")
        if not os.path.isfile(performance_file):
            logging.info("Performance file is not exited, file={}".format(performance_file))
            return []
        with open(performance_file, 'r') as f:
            performance = []
            for line in f.readlines():
                line = line.strip()
                if line == "":
                    continue
                data = json.loads(line)
                if isinstance(data, list):
                    data = data[0]
                performance.append(data)
            logging.info("performance={}".format(performance))
        return performance

    def _insert(self, arch):
        """Random insert to arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        idx = np.random.randint(low=0, high=len(arch[0]))
        arch[0].insert(idx, 1)
        idx = np.random.randint(low=0, high=len(arch[1]))
        arch[1].insert(idx, 1)
        return arch

    def _remove(self, arch):
        """Random remove one from arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        # random pop arch[0]
        ones_index = [i for i, char in enumerate(arch[0]) if char == 1]
        idx = random.choice(ones_index)
        arch[0].pop(idx)
        # random pop arch[1]
        ones_index = [i for i, char in enumerate(arch[1]) if char == 1]
        idx = random.choice(ones_index)
        arch[1].pop(idx)
        return arch

    def _swap(self, arch, R):
        """Random swap one in arch code.

        :param arch: input arch code
        :type arch: list
        :return: changed arch code
        :rtype: list
        """
        while True:
            not_ones_index = [i for i, char in enumerate(arch) if char != 1]
            idx = random.choice(not_ones_index)
            r = random.randint(1, R)
            direction = -r if random.random() > 0.5 else r
            try:
                arch[idx], arch[idx + direction] = arch[idx + direction], arch[
                    idx]
                break
            except Exception:
                continue
        return arch

    def is_valid(self, arch):
        """Check if valid for arch code.

        :param arch: input arch code
        :type arch: list
        :return: if current arch is valid
        :rtype: bool
        """
        return True

    def save_output(self, output_path):
        """Save result to output_path.

        :param output_path: the result save output_path
        :type: str
        """
        try:
            self.pareto_front.sieve_board.to_csv(
                os.path.join(output_path, 'nas_score_board.csv'),
                index=None, header=True)
        except Exception as e:
            logging.error("write nas_score_board.csv error:{}".format(str(e)))
        try:
            pareto_dict = self.pareto_front.get_pareto_front()
            if len(pareto_dict) > 0:
                id = list(pareto_dict.keys())[0]
                net_desc = pareto_dict[id]
                net_desc = self.codec.decode(net_desc)
                with open(os.path.join(output_path, self._best_desc_file), 'w') as fp:
                    json.dump(net_desc, fp)
        except Exception as e:
            logging.error("write best model error:{}".format(str(e)))
