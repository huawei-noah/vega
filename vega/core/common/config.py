# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Load public configuration from yaml or py file and convert dictionary types to objects."""

import yaml
import json
import copy
import os.path as osp
import sys
from importlib import import_module


class Config(dict):
    """A Config class is inherit from dict.

    Config class can parse arguments from a config file of yaml, json or pyscript.
    :param args: tuple of Config initial arguments
    :type args: tuple of str or dict
    :param kwargs: dict of Config initial argumnets
    :type kwargs: dict
    """

    def __init__(self, *args, **kwargs):
        """Init config class with multiple config files or dictionary."""
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('.yaml') or arg.endswith('.yml'):
                    with open(arg) as f:
                        raw_dict = yaml.load(f, Loader=yaml.FullLoader)
                        _dict2config(self, raw_dict)
                elif arg.endswith('.py'):
                    module_name = osp.basename(arg)[:-3]
                    config_dir = osp.dirname(arg)
                    sys.path.insert(0, config_dir)
                    mod = import_module(module_name)
                    sys.path.pop(0)
                    raw_dict = {
                        name: value
                        for name, value in mod.__dict__.items()
                        if not name.startswith('__')
                    }
                    sys.modules.pop(module_name)
                    _dict2config(self, raw_dict)
                elif arg.endswith(".json"):
                    with open(arg) as f:
                        raw_dict = json.load(f)
                        _dict2config(self, raw_dict)
                else:
                    raise Exception('config file must be yaml or py')
            elif isinstance(arg, dict):
                _dict2config(self, arg)
            else:
                raise TypeError('args is not dict or str')
        if kwargs:
            _dict2config(self, kwargs)

    def __call__(self, *args, **kwargs):
        """Call config class to return a new Config object.

        :return: a new Config object.
        :rtype: Config

        """
        return Config(self, *args, **kwargs)

    def __setstate__(self, state):
        """Set state is to restore state from the unpickled state values.

        :param dict state: the `state` type should be the output of
             `__getstate__`.

        """
        _dict2config(self, state)

    def __getstate__(self):
        """Return state values to be pickled.

        :return: change the Config to a dict.
        :rtype: dict

        """
        d = dict()
        for key, value in self.items():
            if type(value) is Config:
                value = value.__getstate__()
            d[key] = value
        return d

    def __getattr__(self, key):
        """Get a object attr by its `key`.

        :param str key: the name of object attr.
        :return: attr of object that name is `key`.
        :rtype: attr of object.

        """
        return self[key]

    def __setattr__(self, key, value):
        """Get a object attr `key` with `value`.

        :param str key: the name of object attr.
        :param value: the `value` need to set to target object attr.
        :type value: attr of object.

        """
        self[key] = value

    def __delattr__(self, key):
        """Delete a object attr by its `key`.

        :param str key: the name of object attr.

        """
        del self[key]

    def __deepcopy__(self, memo):
        """After `copy.deepcopy`, return a Config object.

        :param dict memo: same to deepcopy `memo` dict.
        :return: a deep copyed self Config object.
        :rtype: Config object

        """
        return Config(copy.deepcopy(dict(self)))


def _dict2config(config, dic):
    """Convert dictionary to config.

    :param Config config: config
    :param dict dic: dictionary

    """
    if isinstance(dic, dict):
        for key, value in dic.items():
            if isinstance(value, dict):
                sub_config = Config()
                dict.__setitem__(config, key, sub_config)
                _dict2config(sub_config, value)
            else:
                config[key] = dic[key]
