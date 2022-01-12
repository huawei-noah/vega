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
