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
from vega.core.common.utils import update_dict
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc


class QuantCodec(Codec):
    """Codec class of QuantEA.

    :param codec_name: name of codec
    :type codec_name: string
    :param search_space: search space
    :type search_space: dict
    """

    def __init__(self, codec_name, search_space):
        super(QuantCodec, self).__init__(codec_name, search_space)
        self.search_space = search_space.search_space

    def decode(self, code):
        """Decode the code.

        :param code: code of network
        :type code: list
        :return: network desc
        :rtype: NetworkDesc
        """
        length = len(code)
        desc = {
            "backbone":
            {
                "nbit_w_list": code[: length // 2],
                "nbit_a_list": code[length // 2:]
            }
        }
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return NetworkDesc(desc)
