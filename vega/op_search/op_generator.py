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

"""This is a class for OpGenerator."""
import random
import copy
import logging
import numpy as np
import yaml
import pandas as pd
from vega.common.dag import DAG
from .ops import filter_rules, init_dict, unary_ops, binary_ops, constant_nodes, MAX_LEN_OF_FORMULA
from .utils import get_upstreams, dag2compute


class OpGenerator():
    """Define OpGenerator."""

    def __init__(self, population_num=50, max_sample=20000, search_space=None):
        self.max_sample = max_sample
        self.population_num = population_num
        self.sample_count = 0
        self.sieve_columns = ['sample_id', 'code', 'fitness']
        self.population = pd.DataFrame(columns=self.sieve_columns)
        self.all_samples = pd.DataFrame(columns=self.sieve_columns)
        self.oldest_index = 0
        if search_space is not None:
            with open(search_space, "r", encoding="utf-8") as f:
                res = yaml.safeload(f, Loader=yaml.FullLoader)
            self.unary_ops = res['unary_ops']
            self.binary_ops = res['binary_ops']
        else:
            self.unary_ops = unary_ops
            self.binary_ops = binary_ops

    def is_finished(self):
        """Check if the search is finished."""
        return self.sample_count >= self.max_sample

    def search(self):
        """Generate a sample."""
        if len(self.population) <= self.population_num:
            sample = self.random_search()
        else:
            sample = self.ea_search()
        if filter_rules(sample) and not self._check_duplication(sample) and self._check_dag_valid(sample):
            self.sample_count += 1
            self.all_samples = self.all_samples.append(
                [{"sample_id": self.sample_count, "code": sample}], ignore_index=True)
            sample = self._resample(sample)
            return sample
        else:
            return None

    def _resample(self, old_sample):
        new_sample = copy.deepcopy(old_sample)
        random_input = np.random.random([1, 3, 224, 224]).astype(np.float32) * 20 - 10
        dag = DAG()
        dag.from_dict(new_sample)
        res = dag2compute(dag, random_input)
        if isinstance(res, np.ndarray) and res.max() > 1e5:
            logging.debug("Resample.")
            last_node = get_upstreams(dag, 'out')[0]
            const_op = self.add_subscript(new_sample, 'const1')
            add_op = self.add_subscript(new_sample, 'add')
            div_op = self.add_subscript(new_sample, 'div')
            new_sample[const_op] = [add_op, div_op]
            init_downstream = new_sample[last_node]
            for i in range(len(init_downstream)):
                if init_downstream[i] == 'out':
                    init_downstream[i] = add_op
                    break
            new_sample[last_node] = init_downstream
            new_sample[add_op] = [div_op]
            new_sample[div_op] = ['out']
        return new_sample

    def _check_duplication(self, sample):
        """Check if the code has been sampled."""
        if len(self.all_samples) > 0 and sample in list(self.all_samples.loc[:, 'code']):
            return True
        else:
            return False
        return False

    def _check_equivalence(self, fitness):
        """Check if the sample is equivalence."""
        if len(self.population) > 0 and np.isclose(fitness, list(self.population.loc[:, 'fitness']), atol=1e-3,
                                                   rtol=1e-3).any():
            return True

        else:
            return False
        return False

    def _check_dag_valid(self, sample):
        """Check if the dag is valid or not."""
        try:
            dag = DAG()
            dag.from_dict(sample)
            if dag.size() <= MAX_LEN_OF_FORMULA:
                return True
            else:
                return False
        except Exception:
            logging.debug('The sample {} is not a valid dag.'.format(sample))
            return False

    def random_search(self):
        """Generate a sample by random search."""
        if len(self.all_samples) > 0:
            index = random.randint(0, len(self.all_samples) - 1)
            default_ind = self.all_samples.iat[index, 1]
        else:
            default_ind = init_dict
        ind = copy.deepcopy(default_ind)
        seed = random.randint(0, 2)
        if seed == 0:
            ind = self.insert(ind)

        elif seed == 1:
            ind = self.remove(ind)

        else:
            ind = self.swap(ind)
        return ind

    def ea_search(self):
        """Generate a sample by ea search."""
        select_ind = self.select_parent()
        ind = copy.deepcopy(select_ind)
        seed = random.randint(0, 2)
        if seed == 0:
            ind = self.insert(ind)

        elif seed == 1:
            ind = self.remove(ind)

        else:
            ind = self.swap(ind)
        return ind

    def update_fitness(self, num_id, sample, fitness):
        """Update the fitness of the populaiton."""
        if len(self.population) <= self.population_num:
            self.population = self.population.append(
                [{"sample_id": num_id, "code": sample, "fitness": fitness}], ignore_index=True)
        elif fitness < self.population.at[self.oldest_index, 'fitness']:
            if fitness < 0.1:
                self.population.iat[self.oldest_index, 0] = num_id
                self.population.iat[self.oldest_index, 1] = sample
                self.population.iat[self.oldest_index, 2] = fitness
                self.oldest_index = (self.oldest_index + 1) % self.population_num

            elif random.random() < 0.8:
                self.population.iat[self.oldest_index, 0] = num_id
                self.population.iat[self.oldest_index, 1] = sample
                self.population.iat[self.oldest_index, 2] = fitness
                self.oldest_index = (self.oldest_index + 1) % self.population_num

        else:
            prob = max(1 / np.power(fitness / self.population.at[self.oldest_index, 'fitness'], 6), 0.2)
            if random.random() < prob:
                self.population.iat[self.oldest_index, 0] = num_id
                self.population.iat[self.oldest_index, 1] = sample
                self.population.iat[self.oldest_index, 2] = fitness
            self.oldest_index = (self.oldest_index + 1) % self.population_num

    def select_elite(self):
        """Select elite from the population."""
        pop = copy.deepcopy(self.population)
        pop.sort_values(by="fitness", axis=0, ascending=True, inplace=True)
        elite = pop.iloc[0]["code"]
        return elite

    def select_parent(self):
        """Select parent from the population."""
        if random.random() < 0.2:
            return self.select_elite()

        select_num = int(self.population_num * 0.1)
        select_index = random.sample(range(0, self.population_num - 1), select_num)
        select_pop = self.population.iloc[select_index, :]
        select_pop.sort_values(by="fitness", axis=0, ascending=True, inplace=True)
        return select_pop.iloc[0]["code"]

    def insert(self, ind):
        """Insert an operation or node."""
        init_ind = copy.deepcopy(ind)
        logging.debug("Use random insert.")
        all_nodes = self._get_nodes(ind)
        ind_size = len(all_nodes)
        insert_pos = random.randint(0, ind_size - 2)
        insert_node = all_nodes[insert_pos]

        candidate_ops = self.unary_ops + self.binary_ops
        select_op = candidate_ops[random.randint(0, len(candidate_ops) - 1)]
        insert_binary_op = True if select_op in self.binary_ops else False
        if select_op in all_nodes:
            select_op = self.add_subscript(ind, select_op)
        logging.debug("insert op: {}.".format(select_op))
        dag = DAG()
        dag.from_dict(ind)
        downstreams = dag.next_nodes(node=insert_node)
        if len(downstreams) == 0:
            return ind
        select_pos = random.randint(0, len(downstreams) - 1)
        edges = ind[insert_node]
        tmp = edges[select_pos]
        edges[select_pos] = select_op
        ind[insert_node] = edges
        ind[select_op] = [tmp]

        if insert_binary_op:
            if random.random() < 0.5:
                candidate_insert = [node for node in constant_nodes if node != insert_node]
                extra_insert = candidate_insert[random.randint(0, len(candidate_insert) - 1)]
            else:
                candidate_insert = [node for node in all_nodes if (node != insert_node and node != 'out')]
                if len(candidate_insert) < 1:
                    return init_ind
                index_extra = random.randint(0, len(candidate_insert) - 1)
                extra_insert = candidate_insert[index_extra]
            if extra_insert in all_nodes:
                ind[extra_insert].append(select_op)
            else:
                ind[extra_insert] = [select_op]

        return ind

    def remove(self, ind):
        """Remove an operation or node."""
        logging.debug("Use random remove.")
        init_ind = copy.deepcopy(ind)
        all_ops = self._get_nodes(ind)
        remove_candidate = [node for node in all_ops if
                            (node != 'in' and not node.startswith('const') and node != 'out')]
        if len(remove_candidate) == 0:
            logging.debug("There are no nodes can be removed,remove will not apply. ")
            return init_ind
        remove_pos = random.randint(0, len(remove_candidate) - 1)
        remove_node = remove_candidate[remove_pos]
        logging.debug("removed node: {}.".format(remove_node))
        logging.debug("the individual before remove is:{}.".format(ind))
        dag = DAG()
        dag.from_dict(ind)
        upstreams = get_upstreams(dag, node=remove_node)
        downstreams = dag.next_nodes(node=remove_node)
        if len(upstreams) == 1:
            ind = self._remove_node(ind, remove_node, upstreams[0], downstreams)
        elif len(upstreams) == 2:
            leaf_index = random.randint(0, 1)
            reserve_index = 1 - leaf_index
            ind = self._remove_node(ind, remove_node, upstreams[reserve_index], downstreams)
            ind = self._remove_node(ind, remove_node, upstreams[leaf_index], [])
        ind.pop(remove_node)
        for downstream in downstreams:
            if downstream.split('-')[0] in self.binary_ops:
                dag.from_dict(ind)
                if len(get_upstreams(dag, downstream)) != 2:
                    logging.debug(
                        "The upstreams of binary_op is smaller than 2, is invalid, remove will not apply.")
                    return init_ind
        logging.debug("the individual after remove is:{}.".format(ind))
        return ind

    def _remove_node(self, ind, removed_node, upstream, downstreams):
        """Remove a node and process one of the upstream."""
        edges = ind[upstream]
        edges.remove(removed_node)
        for downstream in downstreams:
            edges.append(downstream)
        ind[upstream] = edges
        return ind

    def swap(self, ind):
        """Swap an operation or node."""
        logging.debug("Use random swap.")
        init_ind = copy.deepcopy(ind)
        all_nodes = self._get_nodes(ind)
        swap_candidate = [node for node in all_nodes if
                          (node.split('-')[0] != 'in' and not node.startswith('const') and node.split('-')[0] != 'out')]
        if len(swap_candidate) == 0:
            logging.debug("There are no nodes can be swapped,swap will not apply. ")
            return init_ind
        swap_pos = random.randint(0, len(swap_candidate) - 1)
        swap_node = swap_candidate[swap_pos]
        swap_node_type = swap_node.split('-')[0]
        if swap_node_type in self.unary_ops:
            candidate_ops = [op for op in self.unary_ops if op != swap_node_type]
            candidate_node = candidate_ops[random.randint(0, len(candidate_ops) - 1)]
        elif swap_node_type in self.binary_ops:
            candidate_ops = [op for op in self.binary_ops if op != swap_node_type]
            candidate_node = candidate_ops[random.randint(0, len(candidate_ops) - 1)]
        else:
            raise ValueError

        if candidate_node in all_nodes:
            candidate_node = self.add_subscript(ind, candidate_node)

        dag = DAG()
        dag.from_dict(ind)
        upstreams = get_upstreams(dag, swap_node)
        for upstream in upstreams:
            edges = ind[upstream]
            index = 0
            for edge in edges:
                if edge == swap_node:
                    break
                index += 1
            edges[index] = candidate_node
            ind[upstream] = edges
        ind[candidate_node] = ind[swap_node]
        ind.pop(swap_node)

        return ind

    def add_subscript(self, ind, select_op):
        """Add subscript to the node if the same op is used one more time. e.g. exp, exp-1."""
        repeated_op = [name for name in ind.keys() if name.split('-')[0] == select_op]
        exist_subscript = [int(name.split('-')[1]) for name in repeated_op if '-' in name]
        if len(exist_subscript) == 0:
            exist_subscript = [0]
        select_op += "-"
        select_op += str(max(exist_subscript) + 1)
        return select_op

    def _get_nodes(self, ind):
        dag = DAG()
        dag.from_dict(ind)
        return dag.topological_sort()
