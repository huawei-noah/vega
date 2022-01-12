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
import numpy as np

from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class AdelaideCodec(Codec):
    """Codec of the Adelaide search space."""

    def __init__(self, search_space=None, **kwargs):
        """Construct the AdelaideCodec class.

        :param codec_name: name of the codec
        :param search_space: Search space of the codec
        """
        super(AdelaideCodec, self).__init__(search_space, **kwargs)

    def encode(self, sample_desc):
        """Add the encoded string to decoded config.

        :param sample_desc: config of decoded structure, which contains the key "config"
        :return: config of encoded structure, which contains the key "code"
        """

        def num_to_str(number):
            return np.base_repr(number, 36)

        def list_to_str(data):
            if isinstance(data, list):
                return ''.join([list_to_str(x) for x in data])
            else:
                return num_to_str(data)

        config = sample_desc.config
        sample_desc['code'] = "_{}".format(list_to_str(config))
        return sample_desc

    def decode(self, sample):
        """Add the block structure to encoded config.

        :param sample: config of encoded structure, which contains the key "code"
        :return: config of decoded structure, which contains the key "config"
        """

        def str_to_num(number):
            return int(number, 36)

        if 'code' not in sample:
            raise ValueError('No code to decode in sample: {}'.format(sample))
        lst = list(map(str_to_num, sample.code[1:]))
        sample['config'] = [[lst[0], lst[1:5], lst[5:9], lst[9:13]], [lst[13:15], lst[15:17], lst[17:19]]]
        return sample
