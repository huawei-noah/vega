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

"""Dictionary based configuration."""

from typing import Dict, Optional, Any
import copy
import logging
import yaml


logger = logging.getLogger('modnas.config')


def merge_config(src: Any, dest: Any, extend_list: bool = True, overwrite: bool = True) -> Any:
    """Return merged config."""
    if isinstance(src, dict) and isinstance(dest, dict):
        for k, v in dest.items():
            if k not in src:
                src[k] = v
                logger.debug('merge_config: add key %s' % k)
            else:
                src[k] = merge_config(src[k], v, extend_list, overwrite)
    elif isinstance(src, list) and isinstance(dest, list) and extend_list:
        logger.debug('merge_config: extend list: %s + %s' % (src, dest))
        src.extend(dest)
    elif overwrite:
        logger.debug('merge_config: overwrite: %s -> %s' % (src, dest))
        src = dest
    return src


def load_config_file(filename: str) -> Dict[str, Any]:
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

    def __init__(self, dct: Optional[Dict] = None, file: Optional[str] = None) -> None:
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

    def to_dict(self) -> Dict[str, Any]:
        """Return dict converted from Config."""
        dct = {}
        for k, v in self.items():
            if isinstance(v, Config):
                v = v.to_dict()
            dct[k] = v
        return dct

    def __deepcopy__(self, memo: Dict[int, Any]) -> Any:
        """Return deepcopy."""
        return Config(copy.deepcopy(dict(self)))

    def __str__(self) -> str:
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
    def set_value(config: Any, key: str, value: Any) -> None:
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
    def apply(config: Any, spec: Any) -> None:
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
    def load(conf: Any) -> Any:
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
