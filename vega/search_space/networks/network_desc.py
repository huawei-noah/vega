# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined NetworkDesc."""
import hashlib
import logging
import json
from copy import deepcopy
import torch.nn as nn
from vega.core.common import Config
from .network_factory import NetworkFactory, NetTypes, NetTypesMap


class NetworkDesc(object):
    """NetworkDesc."""

    def __init__(self, desc):
        """Init NetworkDesc."""
        self._desc = Config(deepcopy(desc))
        self._model_type = None
        self._model_name = None

    def to_model(self):
        """Transform a NetworkDesc to a special model."""
        if 'modules' not in self._desc:
            logging.warning('network=%s does not have key modules.', self._desc)
            return None
        networks = []
        module_types = self._desc.get('modules')
        if self._desc.get('type') != 'Network':
            for module_type in module_types:
                # TODO: unify name and type
                network = self.to_coarse_network(module_type)
                networks.append(network)
            if len(networks) == 1:
                return networks[0]
            else:
                return nn.Sequential(*networks)
        else:
            from .pytorch.network import Network
            is_freeze = self._desc.pop('is_freeze') if 'is_freeze' in self._desc else None
            condition = self._desc.pop('condition') if 'condition' in self._desc else None
            out_list = self._desc.pop('out_list') if 'out_list' in self._desc else None
            for module_type in module_types:
                if module_type == 'process_model':
                    if self._desc['process_model']['condition'] == 'quant':
                        nbit_w_list = self._desc['process_model']['nbit_w_list']
                        nbit_a_list = self._desc['process_model']['nbit_w_list']
                        return Network(networks, is_freeze, 'quant', out_list,
                                       nbit_w_list=nbit_w_list, nbit_a_list=nbit_a_list)
                    elif self._desc['process_model']['condition'] == 'prune':
                        chn_node_mask = self._desc['process_model']['chn_node_mask']
                        chn_mask = self._desc['process_model']['chn_mask']
                        path = self._desc['process_model']['path']
                        return Network(networks, is_freeze, 'prune', out_list,
                                       chn_node_mask=chn_node_mask, chn_mask=chn_mask, path=path)
                else:
                    if isinstance(module_type, list):
                        module_type = module_type[0]
                    try:
                        network = self.to_fine_grained_network(module_type)
                        if network:
                            networks.append(network)
                    except Exception as ex:
                        logging.error("Failed to create Network={}, error message={}".format(module_type, str(ex)))
                        raise ex
            return Network(networks, is_freeze, condition, out_list)

    def to_fine_grained_network(self, network_name):
        """Create network form network desc by name.

        :param network_name:
        :return:
        """
        network_desc = deepcopy(self._desc.get(network_name))
        if not network_desc or not ('type' in network_desc):
            raise KeyError('module descript does not have key {}'.format(network_name))
        if network_desc.get('type') == 'Network':
            network = NetworkDesc(network_desc).to_model()
        else:
            module_type = network_desc.pop('type')
            if 'is_freeze' in network_desc:
                network_desc.pop('is_freeze')
            network_cls = NetworkFactory.get_network(NetTypes.Operator, module_type)
            network = network_cls(**network_desc) if network_desc else network_cls()
        return network

    def to_coarse_network(self, module_type):
        """Create coarse network by module type."""
        module_desc = deepcopy(self._desc.get(module_type))
        if 'name' not in module_desc:
            raise KeyError('module descript does not have key {name}')
        module_name = module_desc.get('name')
        module_type = NetTypesMap[module_type.lower()]
        if self._model_name is None:
            self._model_name = module_name
        if self._model_type is None:
            self._model_type = module_type
        network_cls = NetworkFactory.get_network(module_type, module_name)
        if network_cls is None:
            raise Exception("Network type error, module name: {}, module_type: {}".format(module_type, module_name))
        if module_type == NetTypes.TORCH_VISION_MODEL:
            args = deepcopy(module_desc)
            del args["name"]
            from vega.model_zoo.torch_vision_model import set_torch_home
            set_torch_home()
            network = network_cls(**args)
        else:
            network = network_cls(module_desc)
        return network

    @property
    def md5(self):
        """MD5 value of network description."""
        return self.get_md5(self._desc)

    @classmethod
    def get_md5(cls, desc):
        """Get desc's short md5 code.

        :param desc: network description.
        :type desc: str.
        :return: short MD5 code.
        :rtype: str.

        """
        _desc = deepcopy(desc)
        keys = ["modules"] + _desc["modules"]
        _desc = {key: _desc[key] for key in keys}
        code = hashlib.md5(json.dumps(_desc, sort_keys=True).encode('utf-8')).hexdigest()
        return code[:8]

    @property
    def model_type(self):
        """Return model type."""
        return self._model_type

    @property
    def model_name(self):
        """Return model name."""
        return self._model_name
