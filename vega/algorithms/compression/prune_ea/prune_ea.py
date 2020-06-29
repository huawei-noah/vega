# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Evolution Algorithm used to prune model."""
import numpy as np
import random
import pandas as pd
from vega.search_space.search_algs import SearchAlgorithm
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.search_space.search_algs.nsga_iii import SortAndSelectPopulation
from vega.search_space.codec import Codec
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class PruneEA(SearchAlgorithm):
    """Class of Evolution Algorithm used to Prune Example.

    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space):
        super(PruneEA, self).__init__(search_space)
        self.length = self.policy.length
        self.num_individual = self.policy.num_individual
        self.num_generation = self.policy.num_generation
        self.x_axis = 'flops'
        self.y_axis = 'acc'
        self.random_models = self.policy.random_models
        self.codec = Codec(self.cfg.codec, search_space)
        self.random_count = 0
        self.ea_count = 0
        self.ea_epoch = 0
        self.step_path = FileOps.join_path(self.local_output_path, self.cfg.step_name)
        self.pd_file_name = FileOps.join_path(self.step_path, "performance.csv")
        self.pareto_front_file = FileOps.join_path(self.step_path, "pareto_front.csv")
        self.pd_path = FileOps.join_path(self.step_path, "pareto_front")
        FileOps.make_dir(self.pd_path)

    def get_pareto_front(self):
        """Get pareto front from remote result file."""
        with open(self.pd_file_name, "r") as file:
            df = pd.read_csv(file)
        fitness = df[[self.x_axis, self.y_axis]].values.transpose()
        # acc2error
        fitness[1, :] = 1 - fitness[1, :]
        _, _, selected = SortAndSelectPopulation(fitness, self.num_individual)
        result = df.loc[selected, :]
        if self.ea_count % self.num_individual == 0:
            file_name = "{}_epoch.csv".format(
                str(self.ea_epoch))
            pd_result_file = FileOps.join_path(self.pd_path, file_name)
            with open(pd_result_file, "w") as file:
                result.to_csv(file, index=False)
            with open(self.pareto_front_file, "w") as file:
                result.to_csv(file, index=False)
            self.ea_epoch += 1
        return result

    def crossover(self, ind0, ind1):
        """Cross over operation in EA algorithm.

        :param ind0: individual 0
        :type ind0: list of int
        :param ind1: individual 1
        :type ind1: list of int
        :return: new individual 0, new individual 1
        :rtype: list of int, list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind0.copy()
        b_copy = ind1.copy()
        a_copy[start_idx: end_idx] = ind1[start_idx: end_idx]
        b_copy[start_idx: end_idx] = ind0[start_idx: end_idx]
        return a_copy, b_copy

    def mutatation(self, ind):
        """Mutate operation in EA algorithm.

        :param ind: individual
        :type ind: list of int
        :return: new individual
        :rtype: list of int
        """
        two_idxs = np.random.randint(0, self.length, 2)
        start_idx, end_idx = np.min(two_idxs), np.max(two_idxs)
        a_copy = ind.copy()
        for k in range(start_idx, end_idx):
            a_copy[k] = 1 - a_copy[k]
        return a_copy

    def search(self):
        """Search one NetworkDesc from search space.

        :return: search id, network desc
        :rtype: int, NetworkDesc
        """
        if self.random_count < self.random_models:
            self.random_count += 1
            return self.random_count, self._random_sample()
        pareto_front_results = self.get_pareto_front()
        pareto_front = pareto_front_results["encoding"].tolist()
        if len(pareto_front) < 2:
            encoding1, encoding2 = pareto_front[0], pareto_front[0]
        else:
            encoding1, encoding2 = random.sample(pareto_front, 2)
        choice = random.randint(0, 1)
        # mutate
        if choice == 0:
            encoding1List = str2list(encoding1)
            encoding_new = self.mutatation(encoding1List)
        # crossover
        else:
            encoding1List = str2list(encoding1)
            encoding2List = str2list(encoding2)
            encoding_new, _ = self.crossover(encoding1List, encoding2List)
        self.ea_count += 1
        net_desc = self.codec.decode(encoding_new)
        return self.random_count + self.ea_count, net_desc

    def _random_sample(self):
        """Choose one sample randomly."""
        individual = []
        prob = random.uniform(0, 1)
        for _ in range(self.length):
            s = random.uniform(0, 1)
            if s > prob:
                individual.append(0)
            else:
                individual.append(1)
        return self.codec.decode(individual)

    def update(self, worker_path):
        """Update PruneEA."""
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_output_path, self.backup_base_path)

    @property
    def is_completed(self):
        """Whether to complete algorithm."""
        return self.ea_epoch >= self.num_generation


def str2list(encoding):
    """Transform a string encoding to two lists."""
    encodingList = [int(x) for x in encoding[1:-1].split(',')]
    return encodingList
