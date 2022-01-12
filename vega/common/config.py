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

"""Load public configuration from yaml or py file and convert dictionary types to objects."""

import sys
import json
import logging
import traceback
import copy
from importlib import import_module
import os.path as osp
import yaml


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
                        raw_dict = yaml.safe_load(f)
                        if "abs_path" in kwargs:
                            file_path = osp.dirname(osp.abspath(arg))
                            self._replace_abs_path(file_path, raw_dict)
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
        if key in self:
            return self[key]
        else:
            raise AttributeError(key)

    def dump_yaml(self, output_file, **kwargs):
        """Dump data to a yaml file."""
        try:
            with open(output_file, "w") as f:
                data = json.loads(json.dumps(self))
                yaml.dump(data, f, indent=4, Dumper=SafeDumper, sort_keys=False, **kwargs)
        except Exception as e:
            logging.error(f"Failed to dump config to file: {output_file}. error message: {e}")
            logging.debug(traceback.format_exc())

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

    def _replace_abs_path(self, file_path, raw_dict):
        if isinstance(raw_dict, dict):
            for k, v in raw_dict.items():
                if isinstance(v, dict):
                    self._replace_abs_path(file_path, v)
                elif isinstance(v, str) and v.startswith("./"):
                    raw_dict[k] = osp.join(file_path, v[2:])


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


def build_tree(data):
    """Convert plaint dictionary to a tree dictionary."""
    result = {}
    for key, value in data.items():
        if "." in key:
            _keys = key.split(".")
            _tree = {}
            _tree[_keys[-1]] = value
            _keys.reverse()
            for sub_key in _keys[1:]:
                _tree = {sub_key: _tree}
            branch = result
            for sub_key in key.split("."):
                if sub_key in branch:
                    branch = branch[sub_key]
                    _tree = _tree[sub_key]
                else:
                    branch[sub_key] = _tree[sub_key]
                    break
        else:
            result[key] = value
    return result


class SafeDumper(yaml.SafeDumper):
    """Redefine SafeDumper."""

    def increase_indent(self, flow=False, *args, **kwargs):
        """Fix indent error."""
        return super().increase_indent(flow=flow, indentless=False)
