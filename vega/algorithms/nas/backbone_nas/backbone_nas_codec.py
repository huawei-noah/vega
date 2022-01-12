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

"""Defined BackboneNasCodec."""
import copy
import logging
import random
from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class BackboneNasCodec(Codec):
    """BackboneNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNasCodec."""
        super(BackboneNasCodec, self).__init__(search_space, **kwargs)

    def encode(self, sample_desc, is_random=False):
        """Encode.

        :param sample_desc: a sample desc to encode.
        :type sample_desc: dict
        :param is_random: if use random to encode, default is False.
        :type is_random: bool
        :return: an encoded sample.
        :rtype: dict

        """
        layer_to_block = {18: (8, [0, 0, 1, 0, 1, 0, 1, 0]),
                          34: (16, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
                          50: (16, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]),
                          101: (33, [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
                          152: (50, [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])}
        default_count = 3
        base_depth = sample_desc['network.backbone.depth']
        double_channel = sample_desc.get('network.backbone.doublechannel', None)
        down_sample = sample_desc.get('network.backbone.downsample', None)
        code = [[], []]
        if base_depth in layer_to_block:
            if is_random or double_channel != default_count and double_channel is not None:
                rand_index = random.sample(
                    range(0, layer_to_block[base_depth][0]), double_channel)
                code[0] = [0] * layer_to_block[base_depth][0]
                for i in rand_index:
                    code[0][i] = 1
            else:
                code[0] = copy.deepcopy(layer_to_block[base_depth][1])
            if is_random or down_sample != default_count and down_sample is not None:
                rand_index = random.sample(
                    range(0, layer_to_block[base_depth][0]), down_sample)
                code[1] = [0] * layer_to_block[base_depth][0]
                for i in rand_index:
                    code[1][i] = 1
            else:
                code[1] = copy.deepcopy(layer_to_block[base_depth][1])
        sample = copy.deepcopy(sample_desc)
        sample['code'] = code
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
        if "network.backbone.doublechannel" in desc:
            desc["network.backbone.doublechannel"] = code[0]
        if "network.backbone.downsample" in desc:
            desc["network.backbone.downsample"] = code[1]
        logging.info("decode:{}".format(desc))
        return desc
