# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""SuperNet for CARS-DARTS."""
import logging
import torch
from vega.search_space.networks import NetworkFactory, NetTypes
from vega.search_space.networks.pytorch.super_network import DartsNetwork
import numpy as np


logger = logging.getLogger(__name__)


@NetworkFactory.register(NetTypes.SUPER_NETWORK)
class CARSDartsNetwork(DartsNetwork):
    """Base CARS-Darts Network of classification.

    :param desc: darts description
    :type desc: Config
    """

    def __init__(self, desc):
        """Init CARSDartsNetwork."""
        super(CARSDartsNetwork, self).__init__(desc)
        k = len(self.desc.normal.genotype)
        num_ops = self.num_ops()
        self.random_alpha = torch.zeros(k, num_ops).cuda()
        self._steps = self.desc.normal.steps
        self.random_individual_node = torch.zeros(self._steps, 2).cuda().long()
        self.random_individual_ops = torch.zeros(self._steps, 2).cuda().long()
        self.len_alpha = k

    def num_paths(self):
        """Get number of path."""
        k_normal = len(self.desc.normal.genotype)
        k_reduce = len(self.desc.reduce.genotype)
        return k_normal + k_reduce

    def num_ops(self):
        """Get number of candidate operations."""
        num_ops = len(self.desc.normal.genotype[0][0])
        return num_ops

    def _random_one_individual(self):
        """Randomly initialize an individual."""
        self.random_individual_node.zero_()
        self.random_individual_ops.zero_()
        num_ops = self.num_ops()
        for i in range(self._steps):
            n = i + 2
            random_idx = np.random.randint(0, n, 2)
            while(np.min(random_idx) == np.max(random_idx)):
                random_idx = np.random.randint(0, n, 2)
            for j in range(2):
                self.random_individual_node[i][j] = random_idx[j].item()
            random_idx = np.random.randint(0, num_ops, 2)
            for j in range(2):
                self.random_individual_ops[i][j] = random_idx[j].item()
        return self.random_individual_node.clone(), self.random_individual_ops.clone()

    def _node_ops_to_alpha(self, node, ops):
        """Calculate alpha according to the connection of nodes and operation.

        :param node: node
        :type node: Tensor
        :param ops: operation
        :type ops: Tensor
        """
        self.random_alpha.zero_()
        start = 0
        n = 2
        for i in range(self._steps):
            end = start + n
            for j in range(2):
                self.random_alpha[start + node[i][j]][ops[i][j]] = 1
            start = end
            n += 1
        return self.random_alpha.clone()

    def _alpha_to_node_ops(self, alpha):
        """Calculate the connection of nodes and operation specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        self.random_individual_node.zero_()
        self.random_individual_ops.zero_()
        start = 0
        for i in range(4):
            n = i + 2
            end = start + n
            idx = torch.argmax(alpha[start:end, :], dim=1)
            cnt = 0
            if torch.nonzero(idx).size(0) > 2:
                logger.error("Illegal alpha.")
            for j in range(n):
                if alpha[start + j, idx[j]] > 0:
                    self.random_individual_node[i][cnt] = j
                    self.random_individual_ops[i][cnt] = idx[j]
                    cnt += 1
            start = end
        return self.random_individual_node.clone(), self.random_individual_ops.clone()

    def random_single_path(self):
        """Randomly sample a path.

        :param depth: the number of paths
        :type depth: int
        :param n_primitives: the number of operations
        :type n_primitives: int
        :return: The randomly sampled path
        :rtype: nn.Tensor
        """
        random_node, random_ops = self._random_one_individual()
        alphas_normal = self._node_ops_to_alpha(random_node, random_ops)
        random_node, random_ops = self._random_one_individual()
        alphas_reduce = self._node_ops_to_alpha(random_node, random_ops)
        alphas = torch.cat([alphas_normal, alphas_reduce], dim=0)
        return alphas

    def forward_random(self, input):
        """Randomly select a path and forward.

        :param input: An input tensor
        :type input: Tensor
        """
        random_node, random_ops = self._random_one_individual()
        alphas_normal = self._node_ops_to_alpha(random_node, random_ops)
        random_node, random_ops = self._random_one_individual()
        alphas_reduce = self._node_ops_to_alpha(random_node, random_ops)
        s0, s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.search:
                if self.desc.network[i + 1] == 'reduce':
                    weights = alphas_reduce
                else:
                    weights = alphas_normal
            else:
                weights = None
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if not self.search:
                if self._auxiliary and i == self._auxiliary_layer:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary and not self.search:
            return logits, logits_aux
        else:
            return logits

    def forward(self, input, alpha):
        """Forward a model that specified by alpha.

        :param input: An input tensor
        :type input: Tensor
        """
        alphas_normal = alpha[:self.len_alpha]
        alphas_reduce = alpha[self.len_alpha:]
        s0, s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if self.search:
                if self.desc.network[i + 1] == 'reduce':
                    weights = alphas_reduce
                else:
                    weights = alphas_normal
            else:
                weights = None
            s0, s1 = s1, cell(s0, s1, weights, self.drop_path_prob)
            if not self.search:
                if self._auxiliary and i == self._auxiliary_layer:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        if self._auxiliary and not self.search:
            return logits, logits_aux
        else:
            return logits

    def crossover(self, alphas_a, alphas_b, ratio=0.5):
        """Crossover for two individuals.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param alphas_b: An individual
        :type alphas_b: nn.Tensor
        :param ratio: Probability to crossover
        :type ratio: float
        :return: The offspring after crossover
        :rtype: nn.Tensor
        """
        # alpha a
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_a[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_a[self.len_alpha:])
        new_alphas_normal_node0 = alphas_normal_node.clone()
        new_alphas_normal_ops0 = alphas_normal_ops.clone()
        new_alphas_reduce_node0 = alphas_reduce_node.clone()
        new_alphas_reduce_ops0 = alphas_reduce_ops.clone()
        # alpha b
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_b[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_b[self.len_alpha:])
        new_alphas_normal_node1 = alphas_normal_node.clone()
        new_alphas_normal_ops1 = alphas_normal_ops.clone()
        new_alphas_reduce_node1 = alphas_reduce_node.clone()
        new_alphas_reduce_ops1 = alphas_reduce_ops.clone()
        # crossover index
        for i in range(new_alphas_normal_node0.size(0)):
            if np.random.rand() < 0.5:
                new_alphas_normal_node0[i] = new_alphas_normal_node1[i].clone()
                new_alphas_normal_ops0[i] = new_alphas_normal_ops1[i].clone()
                new_alphas_reduce_node0[i] = new_alphas_reduce_node1[i].clone()
                new_alphas_reduce_ops0[i] = new_alphas_reduce_ops1[i].clone()
        alphas_normal = self._node_ops_to_alpha(new_alphas_normal_node0, new_alphas_normal_ops0).clone()
        alphas_reduce = self._node_ops_to_alpha(new_alphas_reduce_node0, new_alphas_reduce_ops0).clone()
        alphas = torch.cat([alphas_normal, alphas_reduce], dim=0)
        return alphas

    def mutation(self, alphas_a, ratio=0.5):
        """Mutation for An individual.

        :param alphas_a: An individual
        :type alphas_a: nn.Tensor
        :param ratio: Probability to mutation
        :type ratio: float
        :return: The offspring after mutation
        :rtype: nn.Tensor
        """
        alphas_normal_node, alphas_normal_ops = self._alpha_to_node_ops(alphas_a[:self.len_alpha])
        alphas_reduce_node, alphas_reduce_ops = self._alpha_to_node_ops(alphas_a[self.len_alpha:])
        new_alphas_normal_node0 = alphas_normal_node.clone()
        new_alphas_normal_ops0 = alphas_normal_ops.clone()
        new_alphas_reduce_node0 = alphas_reduce_node.clone()
        new_alphas_reduce_ops0 = alphas_reduce_ops.clone()
        # random alpha
        random_node, random_ops = self._random_one_individual()
        new_alphas_normal_node1 = random_node.clone()
        new_alphas_normal_ops1 = random_ops.clone()
        random_node, random_ops = self._random_one_individual()
        new_alphas_reduce_node1 = random_node.clone()
        new_alphas_reduce_ops1 = random_ops.clone()
        for i in range(new_alphas_normal_node0.size(0)):
            if np.random.rand() < 0.5:
                new_alphas_normal_node0[i] = new_alphas_normal_node1[i].clone()
                new_alphas_normal_ops0[i] = new_alphas_normal_ops1[i].clone()
                new_alphas_reduce_node0[i] = new_alphas_reduce_node1[i].clone()
                new_alphas_reduce_ops0[i] = new_alphas_reduce_ops1[i].clone()
        alphas_normal = self._node_ops_to_alpha(new_alphas_normal_node0, new_alphas_normal_ops0).clone()
        alphas_reduce = self._node_ops_to_alpha(new_alphas_reduce_node0, new_alphas_reduce_ops0).clone()
        alphas = torch.cat([alphas_normal, alphas_reduce], dim=0)
        return alphas
