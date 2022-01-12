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

"""search algorithm for ESR_EA."""

import csv
import logging
import os
import random
from bisect import bisect_right
import numpy as np
import pandas as pd
from vega.common.general import General
from vega.common import FileOps
from vega.common import ClassFactory, ClassType
from vega.core.search_algs import SearchAlgorithm
from .esr_ea_individual import ESRIndividual
from .conf import ESRConfig


@ClassFactory.register(ClassType.SEARCH_ALGORITHM)
class ESRSearch(SearchAlgorithm):
    """Evolutionary search algorithm of the efficient super-resolution."""

    config = ESRConfig()

    def __init__(self, search_space=None, **kwargs):
        """Construct the ESR EA search class.

        :param search_space: config of the search space
        :type search_space: dictionary
        """
        super(ESRSearch, self).__init__(search_space, **kwargs)
        self.individual_num = self.config.policy.num_individual
        self.generation_num = self.config.policy.num_generation
        self.elitism_num = self.config.policy.num_elitism
        self.mutation_rate = self.config.policy.mutation_rate
        self.min_active = self.config.range.min_active
        self.max_params = self.config.range.max_params
        self.min_params = self.config.range.min_params

        self.indiv_count = 0
        self.evolution_count = 0
        self.initialize_pop()
        self.elitism = [ESRIndividual(self.codec) for _ in range(self.elitism_num)]
        self.elit_fitness = [0] * self.elitism_num
        self.fitness_pop = [0] * self.individual_num
        self.fit_state = [0] * self.individual_num

    @property
    def is_completed(self):
        """Tell whether the search process is completed.

        :return: True is completed, or False otherwise
        :rtype: bool
        """
        return self.indiv_count > self.generation_num * self.individual_num

    def update_fitness(self, evals):
        """Update the fitness of each individual.

        :param evals: the evalution
        :type evals: list
        """
        for i in range(self.individual_num):
            self.pop[i].update_fitness(evals[i])

    def update_elitism(self, evaluations):
        """Update the elitism and its fitness.

        :param evaluations: evaluations result
        :type evaluations: list
        """
        popu_all = [ESRIndividual(self.codec) for _ in range(self.elitism_num + self.individual_num)]
        for i in range(self.elitism_num + self.individual_num):
            if i < self.elitism_num:
                popu_all[i].copy(self.elitism[i])
            else:
                popu_all[i].copy(self.pop[i - self.elitism_num])
        fitness_all = self.elit_fitness + evaluations
        sorted_ind = sorted(range(len(fitness_all)), key=lambda k: fitness_all[k])
        for i in range(self.elitism_num):
            self.elitism[i].copy(popu_all[sorted_ind[len(fitness_all) - 1 - i]])
            self.elit_fitness[i] = fitness_all[sorted_ind[len(fitness_all) - 1 - i]]
        logging.info('Generation: {}, updated elitism fitness: {}'.format(self.evolution_count, self.elit_fitness))

    def _log_data(self, net_info_type='active_only', pop=None, value=0):
        """Get the evolution and network information of children.

        :param net_info_type:  defaults to 'active_only'
        :type net_info_type: str
        :param pop: defaults to None
        :type pop: list
        :param value:  defaults to 0
        :type value: int
        :return: log_list
        :rtype: list
        """
        log_list = [value, pop.parameter, pop.flops]
        if net_info_type == 'active_only':
            log_list.append(pop.active_net_list())
        elif net_info_type == 'full':
            log_list += pop.gene.flatten().tolist()
        else:
            pass
        return log_list

    def save_results(self):
        """Save the results of evolution contains the information of pupulation and elitism."""
        _path = FileOps.join_path(self.local_output_path, General.step_name)
        FileOps.make_dir(_path)
        arch_file = FileOps.join_path(_path, 'arch.txt')
        arch_child = FileOps.join_path(_path, 'arch_child.txt')
        sel_arch_file = FileOps.join_path(_path, 'selected_arch.npy')
        sel_arch = []
        with open(arch_file, 'a') as fw_a, open(arch_child, 'a') as fw_ac:
            writer_a = csv.writer(fw_a, lineterminator='\n')
            writer_ac = csv.writer(fw_ac, lineterminator='\n')
            writer_ac.writerow(['Population Iteration: ' + str(self.evolution_count + 1)])
            for c in range(self.individual_num):
                writer_ac.writerow(
                    self._log_data(net_info_type='active_only', pop=self.pop[c],
                                   value=self.pop[c].fitness))

            writer_a.writerow(['Population Iteration: ' + str(self.evolution_count + 1)])
            for c in range(self.elitism_num):
                writer_a.writerow(self._log_data(net_info_type='active_only',
                                                 pop=self.elitism[c],
                                                 value=self.elit_fitness[c]))
                sel_arch.append(self.elitism[c].gene)
        sel_arch = np.stack(sel_arch)
        np.save(sel_arch_file, sel_arch)
        if self.backup_base_path is not None:
            FileOps.copy_folder(self.local_output_path, self.backup_base_path)

    def parent_select(self, parent_num=2, select_type='Tournament'):
        """Select parent from a population with Tournament or Roulette.

        :param parent_num: number of parents
        :type parent_num: int
        :param select_type: select_type, defaults to 'Tournament'
        :type select_type: str
        :return: the selected parent individuals
        :rtype: list
        """
        popu_all = [ESRIndividual(self.codec) for _ in range(self.elitism_num + self.individual_num)]
        parent = [ESRIndividual(self.codec) for _ in range(parent_num)]
        fitness_all = self.elit_fitness
        for i in range(self.elitism_num + self.individual_num):
            if i < self.elitism_num:
                popu_all[i].copy(self.elitism[i])
            else:
                popu_all[i].copy(self.pop[i - self.elitism_num])
                fitness_all = fitness_all + [popu_all[i].fitness]
        fitness_all = np.asarray(fitness_all)
        if select_type == 'Tournament':
            for i in range(parent_num):
                tourn = random.sample(range(len(popu_all)), 2)
                if fitness_all[tourn[0]] >= fitness_all[tourn[1]]:
                    parent[i].copy(popu_all[tourn[0]])
                    fitness_all[tourn[0]] = 0
                else:
                    parent[i] = popu_all[tourn[1]]
                    fitness_all[tourn[1]] = 0
        elif select_type == 'Roulette':
            eval_submean = fitness_all - np.min(fitness_all)
            eval_norm = eval_submean / sum(eval_submean)
            eva_threshold = np.cumsum(eval_norm)
            for i in range(parent_num):
                ran = random.random()
                selec_id = bisect_right(eva_threshold, ran)
                parent[i].copy(popu_all[selec_id])
                eval_submean[selec_id] = 0
                eval_norm = eval_submean / sum(eval_submean)
                eva_threshold = np.cumsum(eval_norm)
        else:
            logging.info('Wrong selection type')
        return parent

    def initialize_pop(self):
        """Initialize the population of first generation."""
        self.pop = [ESRIndividual(self.codec) for _ in range(self.individual_num)]
        for i in range(self.individual_num):
            while self.pop[i].active_num < self.min_active:
                self.pop[i].mutation_using(self.mutation_rate)
            while self.pop[i].parameter > self.max_params or self.pop[i].parameter < self.min_params:
                self.pop[i].mutation_node(self.mutation_rate)

    def get_mutate_child(self, muta_num):
        """Generate the mutated children of the next offspring with mutation operation.

        :param muta_num: number of mutated children
        :type muta_num: int
        """
        for i in range(muta_num):
            if int(self.individual_num / 2) == len(self.elitism):
                self.pop[i].copy(self.elitism[i])
            else:
                self.pop[i].copy(random.sample(self.elitism, 1)[0])
            self.pop[i].mutation_using(self.mutation_rate)
            while self.pop[i].active_num < self.min_active:
                self.pop[i].mutation_using(self.mutation_rate)
            self.pop[i].mutation_node(self.mutation_rate)
            while self.pop[i].parameter > self.max_params or self.pop[i].parameter < self.min_params:
                self.pop[i].mutation_node(self.mutation_rate)

    def get_cross_child(self, muta_num):
        """Generate the children of the next offspring with crossover operation.

        :param muta_num: number of mutated children
        :type muta_num: int
        """
        for i in range(int(self.individual_num / 4)):
            pop_id = muta_num + i * 2
            father, mother = self.parent_select(2, 'Roulette')
            length = np.random.randint(4, int(father.gene.shape[0] / 2))
            location = np.random.randint(0, father.gene.shape[0] - length)
            gene_1 = father.gene.copy()
            gene_2 = mother.gene.copy()
            gene_1[location:(location + length), :] = gene_2[location:(location + length), :]
            gene_2[location:(location + length), :] = father.gene[location:(location + length), :]
            self.pop[pop_id].update_gene(gene_1)
            self.pop[pop_id + 1].update_gene(gene_2)
            while self.pop[pop_id].active_num < self.min_active:
                self.pop[pop_id].mutation_using(self.mutation_rate)
            param = self.pop[pop_id].parameter
            while param > self.max_params or param < self.min_params:
                self.pop[pop_id].mutation_node(self.mutation_rate)
                param = self.pop[pop_id].parameter
            while self.pop[pop_id + 1].active_num < self.min_active:
                self.pop[pop_id + 1].mutation_using(self.mutation_rate)
            param = self.pop[pop_id + 1].parameter
            while param > self.max_params or param < self.min_params:
                self.pop[pop_id + 1].mutation_node(self.mutation_rate)
                param = self.pop[pop_id + 1].parameter

    def reproduction(self):
        """Generate the new offsprings."""
        muta_num = self.individual_num - (self.individual_num // 4) * 2
        self.get_mutate_child(muta_num)
        self.get_cross_child(muta_num)

    def update(self, record):
        """Update function.

        :param local_worker_path: the local path that saved `performance.txt`.
        :type local_worker_path: str
        """
        worker_id = int(record.get("worker_id"))
        performance = float(record.get("rewards"))
        self.fitness_pop[(worker_id - 1) % self.individual_num] = performance
        self.fit_state[(worker_id - 1) % self.individual_num] = 1

    def get_fitness(self):
        """Get the evalutation of each individual.

        :return: a list of evaluations
        :rtype: list
        """
        pd_path = os.path.join(self.local_output_path, 'population_fitness.csv')
        with open(pd_path, "r") as file:
            df = pd.read_csv(file)
        fitness_all = df['PSNR'].values
        fitness = fitness_all[fitness_all.size - self.individual_num:]
        return list(fitness)

    def search(self):
        """Search one random model.

        :return: current number of samples, and the model
        :rtype: int and class
        """
        if self.indiv_count > 0 and self.indiv_count % self.individual_num == 0:
            if np.sum(np.asarray(self.fit_state)) < self.individual_num:
                return
            else:
                self.update_fitness(self.fitness_pop)
                self.update_elitism(self.fitness_pop)
                self.save_results()
                self.reproduction()
                self.evolution_count += 1
                self.fitness_pop = [0] * self.individual_num
                self.fit_state = [0] * self.individual_num
        current_indiv = self.pop[self.indiv_count % self.individual_num]
        indiv_cfg = self.codec.decode(current_indiv)
        self.indiv_count += 1
        logging.info('model parameters:{}, model flops:{}'.format(current_indiv.parameter, current_indiv.flops))
        logging.info('model arch:{}'.format(current_indiv.active_net_list()))
        return self.indiv_count, indiv_cfg

    @property
    def max_samples(self):
        """Get max samples number."""
        return self.generation_num * self.individual_num
