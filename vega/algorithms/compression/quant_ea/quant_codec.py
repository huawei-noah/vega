# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
