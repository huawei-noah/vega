# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""The Adelaide model."""
import copy
import tensorflow as tf
from .adelaide_nn.mobilenetv2_backbone import MobileNetV2Backbone
from .adelaide_nn.micro_decoders import MicroDecoder as Decoder
from .adelaide_nn.layer_factory import resize_bilinear
from vega.search_space.networks.net_utils import NetTypes
from vega.search_space.networks.network_factory import NetworkFactory


@NetworkFactory.register(NetTypes.CUSTOM)
class AdelaideFastNAS(object):
    """Search space of AdelaideFastNAS."""

    def __init__(self, net_desc):
        """Construct the AdelaideFastNAS class.

        :param net_desc: config of the searched structure
        """
        super(AdelaideFastNAS, self).__init__()
        self.desc = copy.deepcopy(net_desc)
        self.backbone_load_path = self.desc['backbone_load_path']
        self.data_format = 'channels_first'

    def __call__(self, input_var, training):
        """Do an inference on AdelaideFastNAS model.

        :param input_var: input tensor
        :return: output tensor
        """
        if self.data_format == 'channels_first':
            input_var = tf.transpose(input_var, [0, 3, 1, 2])
        enc = MobileNetV2Backbone(load_path=self.backbone_load_path, data_format=self.data_format)(input_var, training)
        output = Decoder(**self.desc)(enc, training)
        output = resize_bilinear(output, input_var.get_shape()[2:])
        if self.data_format == 'channels_first':
            output = tf.transpose(output, [0, 2, 3, 1])
        return output
