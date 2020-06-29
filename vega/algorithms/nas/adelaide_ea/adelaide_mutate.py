# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Random search algorithm for AdelaideEA."""
import random
import pandas as pd
import logging
from vega.search_space.search_algs.search_algorithm import SearchAlgorithm
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc
from vega.core.common.class_factory import ClassFactory, ClassType
from vega.core.common.file_ops import FileOps


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class AdelaideMutate(SearchAlgorithm):
    """Search algorithm of the random structures."""

    def __init__(self, search_space=None):
        """Construct the AdelaideMutate class.

        :param search_space: Config of the search space
        """
        super(AdelaideMutate, self).__init__(search_space)
        self.search_space = search_space
        self.codec = Codec(self.cfg.codec, search_space)
        self.max_sample = self.cfg.max_sample
        self.sample_count = 0
        self._copy_needed_file()

    def _copy_needed_file(self):
        if "pareto_front_file" not in self.cfg or self.cfg.pareto_front_file is None:
            raise FileNotFoundError("Config item paretor_front_file not found in config file.")
        init_pareto_front_file = self.cfg.pareto_front_file.replace("{local_base_path}", self.local_base_path)
        self.pareto_front_file = FileOps.join_path(self.local_output_path, self.cfg.step_name, "pareto_front.csv")
        FileOps.make_base_dir(self.pareto_front_file)
        FileOps.copy_file(init_pareto_front_file, self.pareto_front_file)
        if "random_file" not in self.cfg or self.cfg.random_file is None:
            raise FileNotFoundError("Config item random_file not found in config file.")
        init_random_file = self.cfg.random_file.replace("{local_base_path}", self.local_base_path)
        self.random_file = FileOps.join_path(self.local_output_path, self.cfg.step_name, "random.csv")
        FileOps.copy_file(init_random_file, self.random_file)

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        """
        return self.sample_count >= self.max_sample

    def search(self):
        """Search one random model.

        :return: current number of samples, and the model
        """
        search_desc = self.search_space.search_space.custom
        pareto_front_df = pd.read_csv(self.pareto_front_file)
        num_ops = len(search_desc.op_names)
        upper_bounds = [num_ops, 2, 2, num_ops, num_ops, 5, 5, num_ops, num_ops,
                        8, 8, num_ops, num_ops, 4, 4, 5, 5, 6, 6]
        code_to_mutate = random.choice(pareto_front_df['Code'])
        index = random.randrange(len(upper_bounds))
        choices = list(range(upper_bounds[index]))
        choices.pop(int(code_to_mutate[index + 1], 36))
        choice = random.choice(choices)
        code_mutated = code_to_mutate[:index + 1] + str(choice) + code_to_mutate[index + 2:]
        search_desc['code'] = code_mutated
        search_desc['method'] = "mutate"
        logging.info("Mutate from {} to {}".format(code_to_mutate, code_mutated))
        search_desc = self.codec.decode(search_desc)
        self.sample_count += 1
        return self.sample_count, NetworkDesc(self.search_space.search_space)

    def update(self, local_worker_path):
        """Update function.

        :param local_worker_path: Local path that saved `performance.txt`
        :type local_worker_path: str
        """
        pass
