# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Adelaide model."""
import torch.nn.functional as F
import copy
from .adelaide_nn.mobilenetv2_backbone import MobileNetV2Backbone
from .adelaide_nn.micro_decoders import MicroDecoder as Decoder
from vega.search_space.networks.pytorch.network import Network
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.CUSTOM)
class AdelaideFastNAS(Network):
    """Search space of AdelaideFastNAS."""

    def __init__(self, net_desc):
        """Construct the AdelaideFastNAS class.

        :param net_desc: config of the searched structure
        """
        super(AdelaideFastNAS, self).__init__()
        self.desc = copy.deepcopy(net_desc)
        self.encoder = MobileNetV2Backbone(load_path=self.desc.pop("backbone_load_path"))
        self.decoder = Decoder(**self.desc)

    def forward(self, input_var):
        """Do an inference on AdelaideFastNAS model.

        :param input_var: input tensor
        :return: output tensor
        """
        enc = self.encoder(input_var)
        output = self.decoder(enc)
        output = F.interpolate(output, size=input_var.size()[2:], mode='bilinear', align_corners=True)
        return output
