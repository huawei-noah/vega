# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Codec of Prune EA."""
import random
import copy
from vega.core.common import Config
from vega.core.common.utils import update_dict
from vega.search_space.codec import Codec
from vega.search_space.networks import NetworkDesc


class PruneCodec(Codec):
    """Class of Prune EA Codec.

    :param codec_name: name of this codec
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, codec_name, search_space):
        """Init PruneCodec."""
        super(PruneCodec, self).__init__(codec_name, search_space)
        self.search_space = search_space.search_space

    def decode(self, code):
        """Decode the code to Network Desc.

        :param code: input code
        :type code: list of int
        :return: network desc
        :rtype: NetworkDesc
        """
        chn_info = self._code_to_chninfo(code)
        desc = {
            "backbone": chn_info
        }
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return NetworkDesc(desc)

    def _code_to_chninfo(self, code):
        """Transform code to channel info.

        :param code: input code
        :type code: list of int
        :return: channel info
        :rtype: Config
        """
        chn = self.search_space.backbone.base_chn
        chn_node = self.search_space.backbone.base_chn_node
        chninfo = Config()
        if code is None:
            chninfo['chn'] = chn
            chninfo['chn_node'] = chn_node
            chninfo['encoding'] = code
            return chninfo
        chn_mask = []
        chn_node_mask = []
        start_id = 0
        end_id = chn[0]
        for i in range(len(chn)):
            if sum(code[start_id:end_id]) == 0:
                len_mask = len(code[start_id:end_id])
                tmp_mask = [0] * len_mask
                tmp_mask[random.randint(0, len_mask - 1)] = 1
                chn_mask.append(tmp_mask)
            else:
                chn_mask.append(code[start_id:end_id])
            start_id = end_id
            if i + 1 == len(chn):
                end_id += chn_node[0]
            else:
                end_id += chn[i + 1]
        chn = []
        for single_chn_mask in chn_mask:
            chn.append(sum(single_chn_mask))
        for i in range(len(chn_node)):
            if sum(code[start_id:end_id]) == 0:
                len_mask = len(code[start_id:end_id])
                tmp_mask = [0] * len_mask
                tmp_mask[random.randint(0, len_mask - 1)] = 1
                chn_node_mask.append(tmp_mask)
            else:
                chn_node_mask.append(code[start_id:end_id])
            start_id = end_id
            if i + 1 < len(chn_node):
                end_id += chn_node[i + 1]
        chn_node = []
        for single_chn_mask in chn_node_mask:
            chn_node.append(sum(single_chn_mask))
        chninfo['chn'] = chn
        chninfo['chn_node'] = chn_node
        chninfo['chn_mask'] = chn_mask
        chninfo['chn_node_mask'] = chn_node_mask
        chninfo['encoding'] = code
        return chninfo
