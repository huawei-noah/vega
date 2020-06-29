# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Defined NetworkFactory."""
import copy
from .net_utils import NetTypes, NetworkRegistry, NetTypesMap


class NetworkFactory(object):
    """NetworkFactory."""

    __network_registry__ = copy.deepcopy(NetworkRegistry)

    @classmethod
    def register(cls, net_type):
        """Register a network class to NetworkFactory.

        :param net_type: class register as a special `net_type`.
        :type net_type: NetTypes.
        :return: the registered network class.
        :rtype: Class.
        """
        if net_type not in cls.__network_registry__:
            raise ValueError("Cannot register net_type ({})".format(net_type))

        def register_network_cls(t_cls):
            if t_cls.__name__ in cls.__network_registry__[net_type]:
                raise ValueError(
                    "Cannot register duplicate network ({})".format(t_cls.__name__))
            cls.__network_registry__[net_type][t_cls.__name__] = t_cls
            return t_cls

        return register_network_cls

    @classmethod
    def register_custom_cls(cls, net_type, t_cls):
        """Register a network class to NetworkFactory.

        :param net_type: class register as a special `net_type`.
        :type net_type: NetTypes.
        :param t_cls: the registered network class.
        :type t_cls: Class.
        """
        if t_cls.__name__ in cls.__network_registry__[net_type]:
            raise ValueError(
                "Cannot register duplicate network ({})".format(t_cls.__name__))
        cls.__network_registry__[net_type][t_cls.__name__] = t_cls

    @classmethod
    def get_network(cls, net_type, network_name):
        """Get a network class from NetworkFactory.

        :param net_type: class register as a special `net_type`.
        :type net_type: NetTypes
        :param network_name: network class name: `network_name`.
        :type network_name: str
        :return: return a network class from NetworkFactory.
        :rtype: Class or None
        """
        if isinstance(net_type, str):
            net_type = NetTypesMap[net_type]
        if net_type not in cls.__network_registry__:
            raise ValueError("Cannot found net_type ({})".format(net_type))
        if str(network_name) in cls.__network_registry__[net_type]:
            return cls.__network_registry__[net_type][str(network_name)]
        else:
            raise ValueError("Can't get network ({})".format(network_name))

    @classmethod
    def is_exists(cls, net_type, network_name):
        """Determine if this network exists in NetworkFactory.

        :param net_type: class register as a special `net_type`.
        :type net_type: str of NetTypes
        :param network_name: network class name: `network_name`.
        :type network_name: str
        :return: if exists this network.
        :rtype: bool
        """
        if isinstance(net_type, str):
            net_type = NetTypesMap[net_type]
        if net_type not in cls.__network_registry__:
            return False
        if str(network_name) in cls.__network_registry__[net_type]:
            return True
        else:
            return False

    def __init__(self):
        """Init NetworkFactory."""
        pass
