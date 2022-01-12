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

"""The Individual definition of ESR_EA algorithm."""
from bisect import bisect_right
import random
import numpy as np
from .conf import ESRRangeConfig


class ESRIndividual(object):
    """Construct ESR individual class.

    :param net_info: the information list of the network
    :type net_info: list
    :param init: Whether starting only convolution or not
    :type init: bool
    """

    config = ESRRangeConfig()

    def __init__(self, net_info):
        """Construct initialize method."""
        self.net_info = net_info
        self.node_num = self.config.node_num
        self.gene = np.zeros((self.node_num, 2)).astype(int)
        self.init_gene()
        self.active_num = self.using_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()
        self.fitness = 0

    def init_gene(self):
        """Initialize the gene randomly."""
        for gene_ind in range(self.node_num):
            type_prob = self.net_info.func_prob
            self.gene[gene_ind][1] = bisect_right(type_prob, random.random())
            self.gene[gene_ind][0] = np.random.randint(2)

    def using_node_num(self):
        """Get the number of using nodes.

        :return: the number of using nodes
        :rtype: int
        """
        using_num = np.sum(np.asarray(self.gene[:, 0]))
        return using_num

    def active_net_list(self):
        """Get the list of active nodes gene.

        :return: list of active nodes gene
        :rtype: list
        """
        net_list = []
        for using, n in self.gene:
            if using:
                type_str = self.net_info.func_type[n]
                net_list.append(type_str)
        return net_list

    def network_parameter(self):
        """Get the number of parameters in network.

        :return: number of parameters in network
        :rtype: int
        """
        model_info = self.active_net_list()
        f_channel = self.net_info.search_space[self.net_info.search_space['modules'][0]]['G0']
        model_para = 9 * 3 * f_channel + 9 * f_channel * f_channel + 9 * f_channel * f_channel * 4 + 9 * 24 * 3
        Conc_all = 0
        b_in_chan = f_channel
        for i in range(len(model_info)):
            name = model_info[i]
            key = name.split('_')
            b_out_chan = 0
            if i > 0:
                b_in_chan = b_out_chan
            b_grow_rat = int(key[2])
            b_out_chan = int(key[3])
            Conc_all += b_out_chan
            model_para += self.net_info.param_block[
                self.net_info.func_type.index(name)]
            if b_grow_rat != b_in_chan:
                model_para += b_grow_rat * b_in_chan
            if b_out_chan != b_in_chan and b_out_chan != b_grow_rat:
                model_para += b_in_chan * b_out_chan
        model_para += Conc_all * f_channel
        return model_para

    def network_flops(self):
        """Get the FLOPS of network.

        :return: the FLOPS of network
        :rtype: float
        """
        model_info = self.active_net_list()
        f_channel = self.net_info.search_space[self.net_info.search_space['modules'][0]]['G0']
        fpart_para = 9 * 3 * f_channel + 9 * f_channel * f_channel + 9 * f_channel * f_channel * 4
        model_flops = fpart_para * 640 * 360 + 9 * 24 * 3 * 1280 * 720
        Conc_all = 0
        b_in_chan = f_channel
        for i in range(len(model_info)):
            name = model_info[i]
            key = name.split('_')
            b_out_chan = 0
            if i > 0:
                b_in_chan = b_out_chan
            b_grow_rat = int(key[2])
            b_out_chan = int(key[3])
            Conc_all += b_out_chan
            model_flops += self.net_info.flops_block[
                self.net_info.func_type.index(name)]
            if b_grow_rat != b_in_chan:
                model_flops += b_grow_rat * b_in_chan * 640 * 360
            if b_out_chan != b_in_chan and b_out_chan != b_grow_rat:
                model_flops += b_in_chan * b_out_chan * 640 * 360
        model_flops += Conc_all * 32 * 640 * 360
        return model_flops * 2

    def copy(self, source):
        """Copy the individual from another individual.

        :param source: the source Individual
        :type source: Individual Objects
        """
        self.net_info = source.net_info
        self.gene = source.gene.copy()
        self.active_num = source.active_num
        self.active_net = source.active_net
        self.parameter = source.parameter
        self.fitness = source.fitness
        self.flops = source.flops

    def update_fitness(self, performance):
        """Update fitness of one individual.

        :param performance: the score of the evalution
        :type performance: float
        """
        self.fitness = performance

    def update_gene(self, new_gene):
        """Update the gene of individual.

        :param new_gene: new gene
        :type new_gene: list
        """
        self.gene = new_gene.copy()
        self.active_num = self.using_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()

    def mutation_using(self, mutation_rate=0.05):
        """Mutate the using gene of individual.

        :param mutation_rate: the prosibility to mutate, defaults to 0.05
        :type mutation_rate: float
        """
        for node_ind in range(self.node_num):
            if np.random.rand() < mutation_rate:
                self.gene[node_ind][0] = 1 - self.gene[node_ind][0]
        self.active_num = self.using_node_num()
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()

    def mutation_node(self, mutation_rate=0.05):
        """Mutate the active node type of individual.

        :param mutation_rate: the prosibility to mutate, defaults to 0.05
        :type mutation_rate: float
        """
        for node_ind in range(self.node_num):
            if self.gene[node_ind][0] and np.random.rand() < mutation_rate:
                type_prob = self.net_info.func_prob
                self.gene[node_ind][1] = bisect_right(type_prob, random.random())
        self.active_net = self.active_net_list()
        self.parameter = self.network_parameter()
        self.flops = self.network_flops()
