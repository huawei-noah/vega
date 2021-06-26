# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

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
