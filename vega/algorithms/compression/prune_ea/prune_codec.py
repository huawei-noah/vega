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

"""Codec of Prune EA."""
import copy
import random

from vega.common import Config
from vega.common import ClassFactory, ClassType
from vega.common import update_dict
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class PruneCodec(Codec):
    """Class of Prune EA Codec.

    :param codec_name: name of this codec
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space, **kwargs):
        """Init PruneCodec."""
        super(PruneCodec, self).__init__(search_space, **kwargs)
        net_type = self.search_space.backbone.type
        base_depth = self.search_space.backbone.base_depth
        stage = self.search_space.backbone.stage
        net_cls = ClassFactory.get_cls(ClassType.NETWORK, net_type)
        stage_blocks = net_cls._default_blocks[base_depth][:stage]
        self.base_chn, self.base_chn_node = [], []
        channel = self.search_space.backbone.base_channel
        for stage in stage_blocks:
            self.base_chn += [channel] * stage
            self.base_chn_node.append(channel)
            channel *= 2

    def decode(self, code):
        """Decode the code to Network Desc.

        :param code: input code
        :type code: list of int
        :return: network desc
        :rtype: NetworkDesc
        """
        chn_info = self._code_to_chninfo(code)
        desc = {
            "backbone": chn_info,
            "head": {"base_channel": chn_info['chn_node'][-1]}
        }
        desc = update_dict(desc, copy.deepcopy(self.search_space))
        return desc

    def _code_to_chninfo(self, code):
        """Transform code to channel info.

        :param code: input code
        :type code: list of int
        :return: channel info
        :rtype: Config
        """
        chn = copy.deepcopy(self.base_chn)
        chn_node = copy.deepcopy(self.base_chn_node)
        chninfo = Config()
        chninfo['base_chn'] = self.base_chn
        chninfo['base_chn_node'] = self.base_chn_node
        if code is None:
            chninfo['chn'] = chn
            chninfo['chn_node'] = chn_node
            chninfo['encoding'] = code
            return chninfo
        chn_node = [self.search_space.backbone.base_channel] + chn_node
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
        chninfo['chn_node'] = chn_node[1:]
        chninfo['base_channel'] = chn_node[0]
        chninfo['chn_mask'] = chn_mask
        chninfo['chn_node_mask'] = chn_node_mask
        chninfo['encoding'] = code
        return chninfo
