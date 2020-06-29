# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Encode and decode the model config. for SPNAS."""

import copy
import logging
from vega.search_space.codec import Codec
from vega.core.common.config import Config
from vega.algorithms.nas.sp_nas.utils import update_config


class SpNasCodec(Codec):
    """Description for SPNAS model."""

    def __init__(self, codec_name, search_space=None):
        super(SpNasCodec, self).__init__(codec_name, search_space)
        config_template_file = search_space.search_space.config_template_file
        assert config_template_file is not None
        self.config_template = Config(config_template_file)
        if 'epoch' in search_space.search_space.keys():
            self.config_template['total_epochs'] = search_space.search_space.epoch

    def encode(self, block_type, arch, mb_arch):
        """Encode backbone.

        :param block_type: block type of sampled model.
        :type block_type: str
        :param arch: serial-level backone encode.
        :type arch: str
        :param mb_arch: parallel-level backone encode.
        :type mb_arch: str
        :return: an encoded sample.
        :rtype: str
        """
        return '_'.join([block_type, arch, mb_arch])

    def decode(self, code):
        """Generate model structure according to sampler or config file.

        :param code: sample code
        :type code: dict
        :return: config of model structure and sample code
        :rtype: tuple
        """
        config = copy.deepcopy(self.config_template)
        if code is None:
            return config, self._default_code

        config = update_config(config, code)
        logging.info("Decode config:{}".format(config))
        return config, code

    @property
    def _default_code(self):
        """Generate default sample info according to config file.

        :return: Initial sample info
        :rtype: dict
        """
        block_type = self.config_template['model']['backbone']['arch_block']
        groups = self.config_template['model']['backbone']['groups']
        base_width = self.config_template['model']['backbone']['base_width']
        if block_type == 'Bottleneck':
            if groups > 1:
                type_ = 'x(' + str(groups) + 'x' + str(base_width) + 'd)'
            else:
                type_ = 'r'
        else:
            type_ = 'b'
        arch_ = self.config_template['model']['backbone']['layers_arch']
        mb_arch_ = self.config_template['model']['backbone']['mb_arch']
        arch = '_'.join([type_, arch_, mb_arch_])
        code = dict(arch=arch,
                    pre_arch=arch_,
                    pre_worker_id=-1)
        return code
