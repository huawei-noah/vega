# -*- coding: utf-8 -*-

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

"""Utils tools."""

import os
import shutil
import sys
import logging
import imp
import random
import socket
from contextlib import contextmanager
from functools import wraps
from copy import deepcopy
import numpy as np


logger = logging.getLogger(__name__)


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


def update_dict(src, dst, exclude=None):
    """Use src dictionary update dst dictionary.

    :param dict src: Source dictionary.
    :param dict dst: Dest dictionary.
    :return: Updated dictionary.
    :rtype: Dictionary
    """
    if exclude is None:
        exclude = ['loss', 'metric', 'lr_scheduler', 'optim', 'model_desc', 'transforms']
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


def init_log(level, log_path="./logs/", log_file="log.txt"):
    """Init logging configuration."""
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
        raise Exception(f"Not supported logging level: {level}")
    fh = logging.FileHandler(os.path.join(log_path, log_file))
    fmt = '%(asctime)s %(levelname)s %(message)s'
    fh.setFormatter(logging.Formatter(fmt))
    logging.getLogger().addHandler(fh)
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
    return fh


def close_log(fh: logging.Handler):
    """Close log."""
    fh.close()
    logging.getLogger().removeHandler(fh)


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


@contextmanager
def switch_directory(dir):
    """Switch to a directory.

    :param dir: directory
    :type dir: str
    """
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    owd = os.getcwd()
    try:
        os.chdir(dir)
        yield dir
    finally:
        os.chdir(owd)


def copy_search_file(srcDir, desDir):
    """Copy files from srcDir to desDir."""
    ls = os.listdir(srcDir)
    for line in ls:
        filePath = os.path.join(srcDir, line)
        if os.path.isfile(filePath):
            shutil.copy(filePath, desDir)


def verify_requires(requires):
    """Verify requires."""
    if requires and isinstance(requires, list):
        failed = []
        for pkg in requires:
            _lower = False
            try:
                __import__(pkg.split("=")[0].replace("<", "").replace(">", "").lower())
            except Exception:
                _lower = True
            try:
                if _lower:
                    __import__(pkg.split("=")[0].replace("<", "").replace(">", ""))
            except Exception:
                failed.append(pkg)
        if failed:
            logger.error("Missing modules: {}".format(failed))
            logger.error("Please run the following command:")
            for pkg in failed:
                logger.error("    pip3 install --user \"{}\"".format(pkg))
            return False
    return True


def verify_platform_pkgs(pkgs: list) -> bool:
    """Verify pytorch, tensorflow or mindspore."""
    failed = []
    for module, pkg in pkgs:
        try:
            __import__(module)
        except Exception:
            failed.append(pkg)
    if failed:
        logger.error("Missing modules: {}".format(failed))
        logger.error("Please run the following command:")
        for pkg in failed:
            logger.error("    pip3 install --user \"{}\"".format(pkg))
        return False
    return True


def remove_np_value(value):
    """Remove np.int64 and np.float32."""
    if value is None:
        return value
    if isinstance(value, np.int64):
        data = int(value)
    elif isinstance(value, np.float32):
        data = float(value)
    elif isinstance(value, dict):
        data = {}
        for key in value:
            data[key] = remove_np_value(value[key])
    elif isinstance(value, list):
        data = []
        for key in range(len(value)):
            data.append(remove_np_value(value[key]))
    elif isinstance(value, tuple):
        data = []
        for key in range(len(value)):
            data.append(remove_np_value(value[key]))
        data = tuple(data)
    elif isinstance(value, np.ndarray):
        data = []
        value = value.tolist()
        for key in range(len(value)):
            data.append(remove_np_value(value[key]))
    else:
        data = value
    return data


def get_available_port(min_port=8000, max_port=9999):
    """Get available port."""
    _sock = socket.socket()
    while True:
        port = random.randint(min_port, max_port)
        try:
            _sock.bind(('', port))
            _sock.close()
            return port
        except Exception:
            logging.debug('Failed to get available port, continue.')
            continue
    return None


def verify_port(port):
    """Verify port."""
    _sock = socket.socket()
    try:
        _sock.bind(('', port))
        _sock.close()
        return True
    except Exception:
        return False
