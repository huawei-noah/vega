# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Encode and decode the model config."""
from vega.search_space.codec import Codec


class SRCodec(Codec):
    """Codec of the MtMSR search space."""

    def __init__(self, codec_name, search_space=None):
        """Construct the SRCodec class.

        :param codec_name: name of the codec
        :param search_space: Search space of the codec
        """
        super(SRCodec, self).__init__(codec_name, search_space)

    def encode(self, sample_desc):
        """Add the encoded string to decoded config.

        :param sample_desc: config of decoded structure, which contains the key "blocks"
        :return: config of encoded structure, which contains the key "code"
        """
        blocks = sample_desc.blocks
        candidates = sample_desc.candidates
        block_to_num = dict(zip(candidates, range(len(candidates))))
        result = list()
        for block in blocks:
            if isinstance(block, list):
                result.append("+{}{}".format(block_to_num[block[0]], block_to_num[block[1]]))
            else:
                result.append(str(block_to_num[block]))
        sample_desc['code'] = ''.join(result)
        return sample_desc

    def decode(self, sample):
        """Add the block structure to encoded config.

        :param sample: config of encoded structure, which contains the key "code"
        :return: config of decoded structure, which contains the key "blocks"
        """
        if 'code' not in sample:
            raise ValueError('No code to decode in sample: {}'.format(sample))
        code = sample.code
        candidates = sample.candidates
        index = 0
        blocks = list()
        while index < len(code):
            if code[index] == '+':
                blocks.append([candidates[int(code[index + 1])], candidates[int(code[index + 2])]])
                index += 3
            else:
                blocks.append(candidates[int(code[index])])
                index += 1
        sample['blocks'] = blocks
        return sample
