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

"""Manage logging states and loggers."""
import os
import time
import copy
import logging
import logging.config
from logging import Logger
from typing import Optional, Dict, Any
from modnas.utils.config import merge_config


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
