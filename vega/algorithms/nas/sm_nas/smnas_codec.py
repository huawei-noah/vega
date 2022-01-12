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

"""Defined BackboneNasCodec."""
from vega.common import ClassType, ClassFactory
from vega.core.search_algs.codec import Codec


@ClassFactory.register(ClassType.CODEC)
class SmNasCodec(Codec):
    """SmNasCodec.

    :param codec_name: name of current Codec.
    :type codec_name: str
    :param search_space: input search_space.
    :type search_space: SearchSpace

    """

    base_arch_code_resnet = {18: '11-21-21-21',
                             34: '111-2111-211111-211',
                             50: '111-2111-211111-211',
                             101: '111-2111-21111111111111111111111-211'}

    base_arch_code_resnext = {50: '111-2111-211111-211',
                              101: '111-2111-21111111111111111111111-211'}

    def __init__(self, search_space=None, **kwargs):
        """Init BackboneNasCodec."""
        super(SmNasCodec, self).__init__(search_space, **kwargs)

    def encode(self, desc):
        """Set code for desc."""
        depth = desc.backbone.backbone.depth
        name = desc.backbone.backbone.type
        if name == 'ResNetDet':
            return self.base_arch_code_resnet[depth]
        elif name == 'ResNeXtDet':
            return self.base_arch_code_resnext[depth]

    def decode(self, code):
        """Decode desc."""
        return {"network.backbone.backbone.code": code}
