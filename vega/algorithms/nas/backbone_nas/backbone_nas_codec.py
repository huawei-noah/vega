# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined BackboneNasCodec."""
import random
import copy
import logging
from vega.search_space.codec import Codec


class BackboneNasCodec(Codec):
    """BackboneNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    def __init__(self, codec_name, search_space=None):
        """Init BackboneNasCodec."""
        super(BackboneNasCodec, self).__init__(codec_name, search_space)

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
                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])}
        default_count = 3
        net_name = list(sample_desc['backbone'].keys())[0]
        net = sample_desc['backbone'][net_name]
        base_depth = net['base_depth']
        double_channel = net['doublechannel']
        down_sample = net['downsample']
        if double_channel != down_sample:
            return None
        code = [[], []]
        if base_depth in layer_to_block:
            if is_random or double_channel != default_count:
                rand_index = random.sample(
                    range(0, layer_to_block[base_depth][0]), double_channel)
                code[0] = [0] * layer_to_block[base_depth][0]
                for i in rand_index:
                    code[0][i] = 1
            else:
                code[0] = copy.deepcopy(layer_to_block[base_depth][1])
            if is_random or down_sample != default_count:
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
        desc = {"modules": []}

        out_channel = 0
        for net_name, net in sample.items():
            desc['modules'].append(net_name)
            tmp_dict = {}
            for sub_net_name, sub_net in net.items():
                tmp_dict = copy.deepcopy(sub_net)
                tmp_dict['name'] = sub_net_name
                break
            tmp_doublechannel = [0]
            if "doublechannel" in tmp_dict:
                tmp_dict["doublechannel"] = code[0]
                tmp_doublechannel = code[0]
            if "base_channel" in tmp_dict:
                if int(tmp_dict["base_depth"]) < 50:
                    out_channel = int(
                        tmp_dict["base_channel"]) * 2**(sum(tmp_doublechannel))
                else:
                    out_channel = int(
                        tmp_dict["base_channel"]) * 4 * 2**(sum(tmp_doublechannel))
            if "downsample" in tmp_dict:
                tmp_dict["downsample"] = code[1]
            if net_name == "head":
                tmp_dict["base_channel"] = out_channel
            else:
                if len(tmp_dict["downsample"]) != len(tmp_dict["doublechannel"]):
                    return None
            desc[net_name] = tmp_dict

        logging.info("decode:{}".format(desc))
        return desc
