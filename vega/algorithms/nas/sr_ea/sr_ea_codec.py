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

"""Encode and decode the model config."""
from vega.core.search_algs.codec import Codec
from vega.common import ClassType, ClassFactory


@ClassFactory.register(ClassType.CODEC)
class SRCodec(Codec):
    """Codec of the MtMSR search space."""

    def __init__(self, search_space=None, **kwargs):
        """Construct the SRCodec class.

        :param codec_name: name of the codec
        :param search_space: Search space of the codec
        """
        super(SRCodec, self).__init__(search_space, **kwargs)

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
