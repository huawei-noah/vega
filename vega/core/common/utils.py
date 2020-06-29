# -*- coding: utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Utils tools."""

import os
import sys
import logging
import imp
from functools import wraps
from copy import deepcopy


def singleton(cls):
    """Set class to singleton class.

    :param cls: class
    :return: instance
    """
    __instances__ = {}

    @wraps(cls)
    def get_instance(*args, **kw):
        """Get class instance and save it into glob list."""
        if cls not in __instances__:
            __instances__[cls] = cls(*args, **kw)
        return __instances__[cls]

    return get_instance


def update_dict(src, dst, exclude=['loss', 'metric', 'lr_scheduler', 'optim', 'model_desc', 'transforms']):
    """Use src dictionary update dst dictionary.

    :param dict src: Source dictionary.
    :param dict dst: Dest dictionary.
    :return: Updated dictionary.
    :rtype: Dictionary
    """
    exclude_keys = exclude or []
    for key in src.keys():
        if key in dst.keys() and key not in exclude_keys:
            if isinstance(src[key], dict):
                dst[key] = update_dict(src[key], dst[key], exclude)
            else:
                dst[key] = src[key]
        else:
            dst[key] = src[key]
    return deepcopy(dst)


def update_dict_with_flatten_keys(desc, flatten_keys):
    """Update dict with flatten keys like `conv.inchannel`.

    :param desc: desc dict
    :param flatten_keys: str
    :return: desc
    """
    if not flatten_keys:
        return desc
    for hyper_param, value in flatten_keys.items():
        dest_param = desc
        dest_key = hyper_param.split('.')[-1]
        for param_key in hyper_param.split('.')[:-1]:
            dest_param = dest_param.get(param_key)
        dest_param[dest_key] = value
    return desc


def init_log(level="info", log_file="log.txt"):
    """Init logging configuration."""
    log_path = "./logs/"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    fmt = '%(asctime)s.%(msecs)d %(levelname)s %(message)s'
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format=fmt,
        datefmt='%Y-%m-%d %H:%M:%S')
    if level == "debug":
        logging.getLogger().setLevel(logging.DEBUG)
    elif level == "info":
        logging.getLogger().setLevel(logging.INFO)
    elif level == "warn":
        logging.getLogger().setLevel(logging.WARN)
    elif level == "error":
        logging.getLogger().setLevel(logging.ERROR)
    elif level == "critical":
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        raise ("Not supported logging level: {}".format(level))
    fh = logging.FileHandler(os.path.join(log_path, log_file))
    fmt = '%(asctime)s %(levelname)s %(message)s'
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)


def lazy(func):
    """Set function as lazy in wrapper.

    :param func: function to be set
    :return: lazy function
    """
    attr_name = "_lazy_" + func.__name__

    def lazy_func(*args, **kwargs):
        if not hasattr(func, attr_name):
            setattr(func, attr_name, func(*args, **kwargs))
        return getattr(func, attr_name)

    return lazy_func


def module_existed(module_name):
    """Test module existed.

    :param module_name: module name.
    :return: True or False
    """
    try:
        imp.find_module(module_name)
        return True
    except ImportError:
        return False
