# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""This is SearchSpace for network."""
from vega.core.common.class_factory import ClassType, ClassFactory
from vega.search_space.fine_grained_space import FineGrainedSpace
from vega.search_space.fine_grained_space.blocks.sr import Shrink_RDB, Group_RDB, Cont_RDB
from vega.search_space.fine_grained_space.conditions import Append
from vega.search_space.fine_grained_space.operators import op, Esrn_Cat


@ClassFactory.register(ClassType.SEARCH_SPACE)
class ERDBLayer(FineGrainedSpace):
    """Create ERDBLayer Searchspace."""

    def constructor(self, arch, G0, kSize):
        """Create ERDBLayer.

        :param arch: arch
        :type arch: dict
        :param G0: G0
        :type G0: G0
        :param kSize: kSize
        :type kSize: int
        """
        b_in_chan = G0
        b_out_chan = 0
        Conc_all = 0
        ERDBs = []
        for i in range(len(arch)):
            name = arch[i]
            key = name.split('_')
            if i > 0:
                b_in_chan = b_out_chan
            b_conv_num = int(key[1])
            b_grow_rat = int(key[2])
            b_out_chan = int(key[3])
            Conc_all += b_out_chan
            if key[0] == 'S':
                ERDBs.append(Shrink_RDB(InChannel=b_in_chan,
                                        OutChannel=b_out_chan,
                                        growRate=b_grow_rat,
                                        nConvLayers=b_conv_num))
            elif key[0] == 'G':
                ERDBs.append(Group_RDB(InChannel=b_in_chan,
                                       OutChannel=b_out_chan,
                                       growRate=b_grow_rat,
                                       nConvLayers=b_conv_num))
            elif key[0] == 'C':
                ERDBs.append(Cont_RDB(InChannel=b_in_chan,
                                      OutChannel=b_out_chan,
                                      growRate=b_grow_rat,
                                      nConvLayers=b_conv_num))
        self.ERBD = Append(*tuple(ERDBs))
        self.cat = Esrn_Cat()
        self.GFF1 = op.Conv2d(in_channels=Conc_all, out_channels=G0, kernel_size=1, stride=1, padding=0)
        self.GFF2 = op.Conv2d(in_channels=G0, out_channels=G0, kernel_size=kSize, stride=1, padding=(kSize - 1) // 2)
