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

"""Defined DblockNasCodec."""
import copy
import random
import numpy as np
from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class DblockNasCodec(Codec):
    """DblockNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    def __init__(self, search_space=None, **kwargs):
        """Init DblockNasCodec."""
        super(DblockNasCodec, self).__init__(search_space, **kwargs)

    def encode(self, sample_desc, is_random=False):
        """Encode.

        :param sample_desc: a sample desc to encode.
        :type sample_desc: dict
        :param is_random: if use random to encode, default is False.
        :type is_random: bool
        :return: an encoded sample.
        :rtype: dict

        """
        op_choices = 7
        channel_choices = 5
        default_marco = '211-2111-211111-211'

        op_num = sample_desc['network.backbone.op_num']
        skip_num = sample_desc['network.backbone.skip_num']
        base_channel = sample_desc['network.backbone.base_channel']

        block_coding = decode_d_block_str(op_choices, channel_choices, op_num, skip_num)

        code = {}
        code['network.backbone.block_coding'] = block_coding
        code['network.backbone.base_channel'] = base_channel
        code['network.backbone.marco_coding'] = default_marco
        sample = {'code': code}

        return sample

    def decode(self, sample):
        """Decode.

        :param sample: input sample to decode.
        :type sample: dict
        :return: return a decoded sample desc.
        :rtype: dict

        """
        if 'code' not in sample:
            raise ValueError('No code to decode in sample:{}'.format(sample))
        code = sample.pop('code')
        desc = copy.deepcopy(sample)
        block_coding = code['network.backbone.block_coding']
        base_channel = code['network.backbone.base_channel']
        marco_coding = code['network.backbone.marco_coding']
        desc['network.backbone.encoding'] = f'{block_coding}_{base_channel}_{marco_coding}'

        return desc


def decode_d_block_str(op_choices, channel_choices, op_num, skip_num):
    """Decode d block str."""

    block_coding = ''
    for i in range(op_num):
        op_index = random.randint(0, op_choices - 1)
        channel_index = random.randint(0, channel_choices - 1)
        if i < op_num - 1:
            block_coding += str(op_index) + str(channel_index)
        else:
            block_coding += str(op_index)

    skip_set = []
    for i in range(op_num + 1):
        for j in range(i + 1, op_num + 1):
            if i == 0 and j == op_num:
                continue
            skip_set.append(str(i) + str(j))

    skip_num = min(skip_num, len(skip_set))
    skip_indexes = list(np.random.permutation(skip_num))
    skip_indexes.sort()

    skip_coding = ''
    for skip_index in skip_indexes:
        if random.randint(0, 1) == 0:
            skip_type = 'a'
        else:
            skip_type = 'c'
        skip_coding += skip_type + skip_set[skip_index]

    block_coding += '-' + skip_coding
    return block_coding
