# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
"""Load config from yaml."""
from copy import deepcopy
from inspect import isclass
from .class_factory import ClassFactory


def load_conf_from_desc(config_cls, desc):
    """Load config class by desc.

    :param desc: desc dict
    :param config_cls: class name
    """
    if not isinstance(desc, dict):
        raise TypeError("desc should be a dict, desc={}".format(desc))
    desc_copy = deepcopy(desc)
    # find the Config object according to desc
    for key, value in desc_copy.items():
        # reference other config objects
        if not hasattr(config_cls, key):
            setattr(config_cls, key, value)
        else:
            # use key as type_name
            sub_config_cls = getattr(config_cls, key)
            # Get config object dynamically according to type
            if not isinstance(sub_config_cls, dict) and hasattr(
                    sub_config_cls, '_class_type') and value and value.get('type'):
                ref_cls = ClassFactory.get_cls(sub_config_cls._class_type, value.type)
                if hasattr(ref_cls, 'config') and ref_cls.config and not isclass(ref_cls.config):
                    sub_config_cls = type(ref_cls.config)
            if not isclass(sub_config_cls) or value is None:
                setattr(config_cls, key, value)
            else:
                if hasattr(sub_config_cls, '_update_all_attrs') and sub_config_cls._update_all_attrs:
                    dict2conf(value, sub_config_cls, is_clear=True)
                else:
                    load_conf_from_desc(sub_config_cls, value)


def dict2conf(dic, conf, is_clear=False):
    """Convert dict to conf.

    :param config: dict
    :param conf: conf obj
    :param is_clear: need to clear all attrs of obj.
    :return:
    """
    if not dic:
        raise ValueError("Config should be a dict.")
    # clear all attrs
    if is_clear:
        for attr_name in dir(conf):
            if attr_name.startswith('_'):
                continue
            delattr(conf, attr_name)

    for key, item in dic.items():
        setattr(conf, key, item)
