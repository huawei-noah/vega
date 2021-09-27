# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Manage logging states and loggers."""
import os
import time
import copy
import logging
import logging.config
from modnas.utils.config import merge_config
from logging import Logger
from typing import Optional, Dict, Any


DEFAULT_LOGGING_CONF = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(message)s',
        }
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        },
        'file': {
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'filename': None,
        }
    },
    'loggers': {
        'modnas': {
            'handlers': ['stream', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}


def get_logger(name: Optional[str] = None) -> Logger:
    """Return logger of given name."""
    root = 'modnas'
    return logging.getLogger(root if name is None else (name if name.startswith(root) else root + '.' + name))


def configure_logging(config: Optional[Dict[str, Any]] = None, log_dir: Optional[str] = None) -> None:
    """Config loggers."""
    config_fn = logging.config.dictConfig
    conf: Dict[str, Any] = copy.deepcopy(DEFAULT_LOGGING_CONF)
    conf['handlers']['file']['filename'] = os.path.join(log_dir or '', '%d.log' % (int(time.time())))
    merge_config(conf, config or {})
    config_fn(conf)


def logged(obj, name=None):
    """Return object with logger attached."""
    obj.logger = get_logger(name or obj.__module__)
    return obj
