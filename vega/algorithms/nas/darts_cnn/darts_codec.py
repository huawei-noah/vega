# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Codec of DARTS."""
import copy
import numpy as np
from vega.core.common import Config
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc


class DartsCodec(Codec):
    """Class of DARTS codec.

    :param codec_name: this codec name
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, codec_name, search_space=None):
        """Init DartsCodec."""
        super(DartsCodec, self).__init__(codec_name, search_space)
        self.darts_cfg = copy.deepcopy(search_space.search_space)
        self.super_net = {'normal': self.darts_cfg.super_network.normal.genotype,
                          'reduce': self.darts_cfg.super_network.reduce.genotype}
        self.super_net = Config(self.super_net)
        self.steps = self.darts_cfg.super_network.normal.steps

    def decode(self, code):
        """Decode the code to Network Desc.

        :param code: input code
        :type code: 2D array of float
        :return: network desc
        :rtype: NetworkDesc
        """
        genotype = self.calc_genotype(code)
        cfg_result = copy.deepcopy(self.darts_cfg)
        cfg_result.super_network.normal.genotype = genotype[0]
        cfg_result.super_network.reduce.genotype = genotype[1]
        return NetworkDesc(cfg_result)

    def calc_genotype(self, arch_param):
        """Parse genotype from arch parameters.

        :param arch_param: arch parameters
        :type arch_param: 2D array of float
        :return: genotype
        :rtype: 2 array of [str, int, int]
        """
        def _parse(weights, genos):
            gene = []
            n = 2
            start = 0
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                G = genos[start:end].copy()
                edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if G[x][k] != 'none'))[:2]  # noqa: E501
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if G[j][k] != 'none':
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append([G[j][k_best], i + 2, j])
                start = end
                n += 1
            return gene
        normal_param = np.array(self.darts_cfg.super_network.normal.genotype)
        reduce_param = np.array(self.darts_cfg.super_network.reduce.genotype)
        geno_normal = _parse(arch_param[0], normal_param[:, 0])
        geno_reduce = _parse(arch_param[1], reduce_param[:, 0])
        return [geno_normal, geno_reduce]
