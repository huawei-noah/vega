# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Modules Config Mapping according to specific backend."""

import copy
import vega
from vega.common.config import Config


class ConfigBackendMapping(object):
    """Config mapping according to backend.

    :param module_type: module type in trainer, 'optim', 'loss' or 'lr_scheduler'
    :type module_type: str
    """

    def __init__(self, type_dict, params_dict):
        """Init config backend mapping."""
        self.type_mapping_dict = copy.deepcopy(type_dict)
        self.params_mapping_dict = copy.deepcopy(params_dict)
        self.backend_type = None
        if vega.is_torch_backend():
            self.backend_type = 'torch'
        elif vega.is_tf_backend():
            self.backend_type = 'tf'
        elif vega.is_ms_backend():
            self.backend_type = 'ms'
        else:
            raise ValueError('Backend type must be torch, tf or ms.')

    def backend_mapping(self, config):
        """Map config to specific backend.

        :param config: original config from config file
        :type config: Config or dict
        :return: config after mapping to backend
        :rtype: Config
        """
        origin_config = Config(copy.deepcopy(config))
        type = origin_config.type

        if type not in self.type_mapping_dict:
            return config
        params = origin_config.get('params', {})
        backend_config = Config()
        backend_config.type = self.type_mapping_dict[type][self.backend_type]
        backend_config.params = Config()

        mapping_params = self.params_mapping_dict.get(type, {})
        for key, value in params.items():
            if key in mapping_params:
                mapping_key = mapping_params[key][self.backend_type]
            else:
                mapping_key = None
            if mapping_key is not None:
                if isinstance(value, dict) and 'type' in value:
                    backend_config.params[mapping_key] = self.backend_mapping(value)
                else:
                    backend_config.params[mapping_key] = value

        return Config(backend_config)
