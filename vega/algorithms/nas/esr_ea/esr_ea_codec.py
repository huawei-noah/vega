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

"""Encode and decode the model config. for ESR."""
import logging
from copy import deepcopy
import numpy as np
from vega.core.search_algs.codec import Codec
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.CODEC)
class ESRCodec(Codec):
    """Codec of the MtMSR search space."""

    def __init__(self, search_space=None, **kwargs):
        """Construct the SRCodec class.

        :param codec_name: name of the codec
        :type codec_name: string
        :param search_space: search space of the codec
        :type search_space: dictionary
        "S_" means that the shrink RDB (SRDB).
        "G_" means that the group RDB (GRDB).
        "C_" means that the contextual RDB (CRDB).
        first number: the number of convolutional layers in a block
        second number: the growth rate of dense connected in a block
        third number: the number of output channel in a block
        """
        super(ESRCodec, self).__init__(search_space, **kwargs)
        self.func_type, self.func_prob = self.get_choices()
        self.param_block = self.get_para_block()
        self.flops_block = self.get_flops_block()
        self.func_type_num = len(self.func_type)

    def get_choices(self):
        """Get search space information.

        :return: the configs of the blocks
        :rtype: list
        """
        model_type = self.search_space['modules'][0]
        block_types = self.search_space[model_type]['block_type']
        block_prob = self.search_space[model_type]['type_prob']
        conv_num = self.search_space[model_type]['conv_num']
        conv_prob = self.search_space[model_type]['conv_prob']
        grow_rate = self.search_space[model_type]['growth_rate']
        growth_prob = self.search_space[model_type]['growth_prob']

        func_type = []
        func_prob = []
        for block_id in range(len(block_types)):
            for conv_id in range(len(conv_num)):
                for grow_id in range(len(grow_rate)):
                    for out_id in range(grow_id, len(grow_rate)):
                        func_type.append("{}_{}_{}_{}".format(
                            block_types[block_id], str(conv_num[conv_id]),
                            str(grow_rate[grow_id]), str(grow_rate[out_id])
                        ))
                        func_prob.append(block_prob[block_id] * conv_prob[conv_id] * growth_prob[grow_id] * growth_prob[
                            out_id])  # noqa: E501
        func_prob = np.cumsum(np.asarray(func_prob) / sum(func_prob))
        return func_type, func_prob

    def get_para_block(self):
        """Get the paras of block.

        :return: the paras of the blocks
        :rtype: int
        """
        block_param = []
        for ind in range(len(self.func_type)):
            key = self.func_type[ind].split('_')
            for i in range(1, len(key)):
                key[i] = int(key[i])
            if key[0] == 'S':
                para = key[2] ** 2 * (9 * key[1] + 0.5 * (2 + key[1]) * (key[1] - 1)) + (key[1] + 1) * key[2] * key[3]
            elif key[0] == 'G':
                para = key[2] ** 2 * 9 * (3.5 + (4 + key[1]) * (key[1] - 3) / 8) + (key[1] + 1) * key[2] * key[3]
            elif key[0] == 'C':
                para = key[2] ** 2 * (9 * key[1] + 0.5 * (2 + key[1]) * (key[1] - 1)) + (key[1] + 1) * key[2] * key[
                    3] / 4
            else:
                logging.info('Wrong block type is used')
            block_param.append(para)
        return block_param

    def get_flops_block(self):
        """Get the Multi-Adds of block.

        :return: the Multi-Adds of the block
        :rtype: int
        """
        block_flops = []
        for ind in range(len(self.func_type)):
            key = self.func_type[ind].split('_')
            for i in range(1, len(key)):
                key[i] = int(key[i])
            if key[0] == 'S':
                flops = (key[2] ** 2 * (9 * key[1] + 0.5 * (2 + key[1]) * (key[1] - 1)) + (key[1] + 1) * key[2] * key[
                    3]) * 640 * 360  # noqa: E501
            elif key[0] == 'G':
                flops = (key[2] ** 2 * 9 * (3.5 + (4 + key[1]) * (key[1] - 3) / 8) + (key[1] + 1) * key[2] * key[
                    3]) * 640 * 360  # noqa: E501
            elif key[0] == 'C':
                flops = (key[2] ** 2 * (9 * key[1] * 3 + 0.5 * (2 + key[1]) * (key[1] - 1))) * 320 * 180 + (
                        (key[1] + 1) * key[2] * key[3] / 4) * 640 * 360  # noqa: E501,E126
            else:
                logging.info('Wrong block type is used')
            block_flops.append(flops)
        return block_flops

    def decode(self, indiv):
        """Add the network structure to config.

        :param indiv: an individual which contains network architecture code
        :type indiv: class
        :return: config of model structure
        :rtype: dictionary
        """
        indiv_cfg = deepcopy(self.search_space)
        model = indiv_cfg['modules'][0]
        indiv_cfg[model]['code'] = indiv.gene.tolist()
        indiv_cfg[model]['architecture'] = indiv.active_net_list()
        return indiv_cfg
