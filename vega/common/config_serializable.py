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

"""Convert class to string."""
import json
import logging
from copy import deepcopy
from inspect import ismethod, isfunction
import numpy as np
from vega.common.check import valid_rule
from .config import Config
from .class_factory import ClassFactory


__all__ = ["ConfigSerializable", "backup_configs"]
logger = logging.getLogger(__name__)
exclude_default = ['type', 'hyperparameters']


class ConfigSerializable(object):
    """Seriablizable config base class."""

    __original__value__ = None

    def to_dict(self):
        """Serialize config to a dictionary."""
        attrs = [attr for attr in dir(self) if not attr.startswith("__")]
        attrs = [attr for attr in attrs if not ismethod(getattr(self, attr)) and not isfunction(getattr(self, attr))]
        attr_dict = {}
        for attr in attrs:
            value = getattr(self, attr)
            if isinstance(value, type) and isinstance(value(), ConfigSerializable):
                value = value().to_dict()
            elif isinstance(value, ConfigSerializable):
                value = value.to_dict()
            attr_dict[attr] = value
        return Config(deepcopy(attr_dict))

    @classmethod
    def from_dict(cls, data, skip_check=True):
        """Restore config from a dictionary or a file."""
        if not data:
            return cls
        if cls.__name__ == "ConfigSerializable":
            return cls
        config = Config(deepcopy(data))
        if not skip_check:
            cls.check_config(config)
        # link config
        if _is_link_config(cls):
            _load_link_config(cls, config)
            return cls
        # normal config
        for attr in config:
            if not hasattr(cls, attr):
                if attr not in exclude_default:
                    logger.debug('{} not in default config ! Please check.'.format(attr))
                setattr(cls, attr, config[attr])
                continue
            class_value = getattr(cls, attr)
            config_value = config[attr]
            if isinstance(class_value, type) and isinstance(config_value, dict):
                setattr(cls, attr, class_value.from_dict(config_value, skip_check))
            else:
                setattr(cls, attr, config_value)
        return cls

    def __repr__(self):
        """Serialize config to a string."""
        return json.dumps(self.to_dict(), cls=NpEncoder)

    @classmethod
    def check_config(cls, config):
        """Check config."""
        valid_rule(cls, config, cls.rules())

    @classmethod
    def rules(cls):
        """Return rules for checking."""
        return {}

    @classmethod
    def backup_original_value(cls, force=False):
        """Backup class original data."""
        if not cls.__original__value__ or force:
            cls.__original__value__ = cls().to_dict()
        return cls.__original__value__

    @classmethod
    def renew(cls):
        """Restore class original data."""
        if cls.__original__value__:
            cls.from_dict(cls.__original__value__)

    @classmethod
    def get_config(cls):
        """Get sub config."""
        return {}


def _is_link_config(_cls):
    return hasattr(_cls, "type") and hasattr(_cls, "_class_type") and \
        hasattr(_cls, "_class_data")


def _load_link_config(_cls, config):
    if not isinstance(config, dict) or "type" not in config:
        logger.error("Failed to unserialize config, class={}, config={}".format(
            str(_cls()), str(config)))
        return None
    class_type = _cls._class_type
    class_name = config["type"]
    if not class_name:
        return None
    if "_class_data" in config:
        # restore config
        class_data = config["_class_data"]
    else:
        # first set config
        class_data = config
    config_cls = _get_specific_class_config(class_type, class_name)
    if config_cls:
        setattr(_cls, "type", class_name)
        if class_data:
            setattr(_cls, "_class_data", config_cls.from_dict(class_data))
            valid_rule(_cls._class_data, config, _cls._class_data.rules())
            for key, sub in _cls._class_data.get_config().items():
                if key in config.keys():
                    valid_rule(sub, config[key], sub.rules())
        else:
            setattr(_cls, "_class_data", None)
    else:
        logger.error("Failed to unserialize config, class={}, config={}".format(
            str(_cls()), str(config)))


def _get_specific_class_config(class_type, class_name):
    specific_class = ClassFactory.get_cls(class_type, class_name)
    if hasattr(specific_class, 'config') and specific_class.config:
        return type(specific_class.config)
    else:
        return None


def backup_configs():
    """Backup all configs."""
    classes = []
    _get_all_config_cls(classes, ConfigSerializable)
    for subclass in classes:
        subclass.backup_original_value()


def _get_all_config_cls(classes, base_class):
    subclasses = base_class.__subclasses__()
    for subclass in subclasses:
        classes.append(subclass)
        _get_all_config_cls(classes, subclass)


class NpEncoder(json.JSONEncoder):
    """Serialize numpy to default."""

    def default(self, obj):
        """Change type of obj."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
