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

"""Codec for searching quantization model."""
import copy
from vega.common import update_dict
from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class QuantCodec(Codec):
    """Codec class of QuantEA.

    :param codec_name: name of codec
    :type codec_name: string
    :param search_space: search space
    :type search_space: dict
    """

    def __init__(self, search_space, **kwargs):
        super(QuantCodec, self).__init__(search_space, **kwargs)

    def decode(self, code):
        """Decode the code.

        :param code: code of network
        :type code: list
        :return: network desc
        :rtype: NetworkDesc
        """
        length = len(code)
        desc = {
            "nbit_w_list": code[: length // 2],
            "nbit_a_list": code[length // 2:]
        }
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return desc
