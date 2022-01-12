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
class PruneMobilenetCodec(Codec):
    """Class of Prune MobileNet EA Codec.

    :param codec_name: name of this codec
    :type codec_name: str
    :param search_space: search space
    :type search_space: SearchSpace
    """

    def __init__(self, search_space, **kwargs):
        """Init PruneMobilenetCodec."""
        super(PruneMobilenetCodec, self).__init__(search_space, **kwargs)
        net_type = self.search_space.backbone.type
        net_cls = ClassFactory.get_cls(ClassType.NETWORK, net_type)
        self.cfgs = net_cls.cfgs

    def decode(self, code):
        """Decode the code to Network Desc.

        :param code: input code
        :type code: list of int
        :return: network desc
        :rtype: NetworkDesc
        """
        chn_info = self._code_to_chninfo(code)
        desc = {
            'backbone': chn_info,
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
        chn_info = Config()
        start_id = 0
        end_id = 0
        if not code:
            return self.cfgs
        chn = copy.deepcopy(self.cfgs)
        chn_mask = []
        for idx, layer in enumerate(self.cfgs):
            if idx == 0:
                pass
            # hidden_dim
            cfg_idx = 1
            end_id += int(layer[cfg_idx])
            if sum(code[start_id:end_id]) == 0:
                len_mask = len(code[start_id:end_id])
                tmp_mask = [0] * len_mask
                tmp_mask[random.randint(0, len_mask - 1)] = 1
                chn_mask.append(tmp_mask)
            else:
                chn_mask.append(code[start_id:end_id])
            chn[idx][cfg_idx] = int(sum(chn_mask[-1]))
            start_id = end_id

            # output_channel
            cfg_idx = 2
            end_id += int(layer[cfg_idx])
            if sum(code[start_id:end_id]) == 0:
                len_mask = len(code[start_id:end_id])
                tmp_mask = [0] * len_mask
                tmp_mask[random.randint(0, len_mask - 1)] = 1
                chn_mask.append(tmp_mask)
            else:
                chn_mask.append(code[start_id:end_id])
            chn[idx][cfg_idx] = int(sum(chn_mask[-1]))
            start_id = end_id

        chn_info['cfgs'] = chn
        chn_info['base_cfgs'] = self.cfgs
        chn_info['chn_mask'] = chn_mask
        chn_info['encoding'] = code
        return chn_info
