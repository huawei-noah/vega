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

"""Codec of DARTS."""
import copy
import numpy as np
from vega.common import Config
from vega.core.search_algs.codec import Codec
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.CODEC)
class DartsCodec(Codec):
    """Class of DARTS codec.

    :param codec_name: this codec name
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space=None, **kwargs):
        """Init DartsCodec."""
        super(DartsCodec, self).__init__(search_space, **kwargs)
        self.darts_cfg = copy.deepcopy(search_space)
        self.super_net = {'cells.normal': self.darts_cfg.super_network.cells.normal.genotype,
                          'cells.reduce': self.darts_cfg.super_network.cells.reduce.genotype}
        self.super_net = Config(self.super_net)
        self.steps = self.darts_cfg.super_network.cells.normal.steps

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
        cfg_result.super_network.search = False
        return cfg_result

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
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if G[x][k] != 'none'))[:2]
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

        normal_param = np.array(self.darts_cfg.super_network.cells.normal.genotype)
        reduce_param = np.array(self.darts_cfg.super_network.cells.reduce.genotype)
        geno_normal = _parse(arch_param[0], normal_param[:, 0])
        geno_reduce = _parse(arch_param[1], reduce_param[:, 0])
        return [geno_normal, geno_reduce]
