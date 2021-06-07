# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Dictionary based configuration."""
# modified from https://github.com/HarryVolek/PyTorch_Speaker_Verification
import yaml
import copy
from . import merge_config


def load_config_file(filename):
    """Load configuration from YAML file."""
    docs = yaml.load_all(open(filename, 'r'), Loader=yaml.SafeLoader)
    config_dict = dict()
    for doc in docs:
        for k, v in doc.items():
            config_dict[k] = v
    return config_dict


class Config(dict):
    """Dictionary based configuration."""

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct=None, file=None):
        super().__init__()
        dct = {} if dct is None else dct
        if file is not None:
            dct = load_config_file(file)
        for key, value in dct.items():
            if hasattr(value, 'items'):
                value = Config(value)
            elif isinstance(value, list):
                for i in range(len(value)):
                    if hasattr(value[i], 'items'):
                        value[i] = Config(value[i])
            self[key] = value
        yaml.add_representer(Config,
                             lambda dumper, data: dumper.represent_mapping('tag:yaml.org,2002:map', data.items()))

    def to_dict(self):
        """Return dict converted from Config."""
        dct = {}
        for k, v in self.items():
            if isinstance(v, Config):
                v = v.to_dict()
            dct[k] = v
        return dct

    def __deepcopy__(self, memo):
        """Return deepcopy."""
        return Config(copy.deepcopy(dict(self)))

    def __str__(self):
        """Return config string."""
        return yaml.dump(dict(self), default_flow_style=False)

    @staticmethod
    def get_value(config, key):
        """Get config value by path."""
        keywords = key.split('.')
        val = config[keywords[0]]
        if len(keywords) == 1:
            return val
        elif val is None:
            raise ValueError('invalid key: {}'.format(keywords[0]))
        return Config.get_value(val, '.'.join(keywords[1:]))

    @staticmethod
    def set_value(config, key, value):
        """Set config value by path."""
        keywords = key.split('.')
        val = config.get(keywords[0], None)
        if len(keywords) == 1:
            config[keywords[0]] = merge_config(val, value)
            return
        if val is None:
            val = Config()
            config[keywords[0]] = val
        Config.set_value(val, '.'.join(keywords[1:]), value)

    @staticmethod
    def apply(config, spec):
        """Apply items to a configuration."""
        if isinstance(spec, dict):
            spec = Config(dct=spec)
            for k, v in spec.items():
                Config.set_value(config, k, v)
        elif isinstance(spec, list):
            for item in spec:
                Config.apply(config, item)
        elif isinstance(spec, str):
            k, v = spec.split('=')
            Config.set_value(config, k, yaml.load(v, Loader=yaml.SafeLoader))
        else:
            raise ValueError('unsupported apply type: {}'.format(type(spec)))

    @staticmethod
    def load(conf):
        """Load configuration."""
        if isinstance(conf, Config):
            config = conf
        elif isinstance(conf, str):
            config = Config(file=conf)
        elif isinstance(conf, dict):
            config = Config(dct=conf)
        else:
            raise ValueError('invalid config type: {}'.format(type(conf)))
        return config
