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

"""Defined DnetNasCodec."""
import copy
import numpy as np
from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class DnetNasCodec(Codec):
    """DnetNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    def __init__(self, search_space=None, **kwargs):
        """Init DnetNasCodec."""
        super(DnetNasCodec, self).__init__(search_space, **kwargs)

    def encode(self, sample_desc, is_random=False):
        """Encode.

        :param sample_desc: a sample desc to encode.
        :type sample_desc: dict
        :param is_random: if use random to encode, default is False.
        :type is_random: bool
        :return: an encoded sample.
        :rtype: dict

        """
        code_length = sample_desc['network.backbone.code_length']
        base_channel = sample_desc['network.backbone.base_channel']
        final_channel = sample_desc['network.backbone.final_channel']
        down_sample = sample_desc['network.backbone.downsample']

        block_coding = sample_desc['block_coding']
        macro_coding = ['1' for _ in range(code_length)]
        channel_times = int(np.log2(final_channel // base_channel))
        while True:
            variant_num = down_sample + channel_times
            variant_positions = np.random.permutation(code_length)[0:variant_num]
            variant_positions.sort()

            down_indexes = np.random.permutation(variant_num)[0:down_sample]
            down_indexes.sort()
            down_positions = variant_positions[down_indexes]

            adjacent_positions = set(down_positions) & set(down_positions + 1)
            if len(adjacent_positions) > 0:
                continue
            break

        variant_positions = list(variant_positions)
        down_positions = list(down_positions)
        for i in variant_positions:
            macro_coding[i] = '2'
            if i in down_positions:
                macro_coding[i] = '-'

        macro_coding = ''.join(macro_coding)

        code = {}
        code['network.backbone.block_coding'] = block_coding
        code['network.backbone.base_channel'] = base_channel
        code['network.backbone.macro_coding'] = macro_coding
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
        macro_coding = code['network.backbone.macro_coding']

        desc['network.backbone.encoding'] = f'{block_coding}_{base_channel}_{macro_coding}'

        return desc
