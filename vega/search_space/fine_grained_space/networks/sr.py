# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
import copy
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space import FineGrainedSpace
from vega.search_space.fine_grained_space.conditions import Sequential, Add, Interpolate
from vega.search_space.fine_grained_space.cells import ERDBLayer
from vega.search_space.fine_grained_space.blocks import UPNet
from vega.search_space.fine_grained_space.networks import MobileNetV3Tiny
from vega.search_space.fine_grained_space.operators import op
from vega.search_space.fine_grained_space.blocks import MicroDecoder


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ESRN(FineGrainedSpace):
    """Create ESRN SearchSpace."""

    def constructor(self, net_desc):
        """Construct the ESRN layer.

        :param net_desc: config of the searched structure
        :type net_desc: dict
        """
        arch = net_desc.architecture
        r = net_desc.scale
        G0 = net_desc.G0
        kSize = 3
        n_colors = 3
        SFENet1 = op.Conv2d(in_channels=n_colors, out_channels=G0, kernel_size=kSize,
                            stride=1, padding=(kSize - 1) // 2)
        SFENet2 = op.Conv2d(in_channels=G0, out_channels=G0, kernel_size=kSize,
                            stride=1, padding=(kSize - 1) // 2)
        ERBDs = ERDBLayer(arch=arch, G0=G0, kSize=kSize)
        seq = Sequential(*tuple([SFENet1, SFENet2, ERBDs]))
        self.merge = Add(SFENet1, seq)
        self.upnet = UPNet(scale=r, G0=G0, kSize=kSize, n_colors=n_colors)


@ClassFactory.register(ClassType.SEARCH_SPACE)
class AdelaideFastNAS(FineGrainedSpace):
    """Search space of AdelaideFastNAS."""

    def constructor(self, net_desc):
        """Construct the AdelaideFastNAS class.

        :param net_desc: config of the searched structure
        """
        desc = copy.deepcopy(net_desc)
        self._nes = desc
        encoder = MobileNetV3Tiny(desc.pop("backbone_load_path"))
        decoder = MicroDecoder(**desc)
        self.block = Interpolate(*tuple([encoder, decoder]))
