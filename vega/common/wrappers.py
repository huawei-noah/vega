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

"""Provide wrapper functions."""

import os
from inspect import signature as sig
from functools import wraps
import vega
from vega.common import ClassFactory, init_log, close_log, General, ClassType


def metric(name=None):
    """Make function as a metrics, use the same params from configuration.

    :param func: source function
    :return: wrapper
    """

    def decorator(func):
        """Provide input param to decorator.

        :param func: wrapper function
        :return: decoratpr
        """
        setattr(func, 'name', name or func.__name__)

        @ClassFactory.register('trainer.metrics')
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Make function as a wrapper."""
            params_sig = sig(func).parameters
            params = {param: value for param, value in kwargs.items() if param in params_sig}
            return func(*args, **params)

        return wrapper

    return decorator


def train_process_wrapper(func):
    """Train process wrapper."""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """Wrap method."""
        log_type = "worker"
        worker_type = getattr(self, "worker_type", None)
        if worker_type is not None:
            worker_type_value = worker_type.value
        else:
            worker_type_value = None
        if worker_type_value == 3:
            log_type = "host_evaluator"
        elif worker_type_value == 5:
            log_type = "device_evaluator"
        fh = init_log(level=General.logger.level,
                      log_file=f"{self.step_name}_{log_type}_{self.worker_id}.log",
                      log_path=self.local_log_path)
        if not getattr(self, "hccl", False):
            pop_rank_envs()
        r = func(self, *args, **kwargs)
        if not getattr(self, "hccl", False):
            restore_rank_envs()
        close_log(fh)
        return r

    return wrapper


_envs = {}


def pop_rank_envs():
    """Pop rank envs."""
    envs = ["RANK_TABLE_FILE", "RANK_SIZE", "RANK_ID"]
    global _envs
    for env in envs:
        if env in os.environ:
            _envs[env] = os.environ[env]
            os.environ.pop(env)


def restore_rank_envs():
    """Restore rank envs."""
    global _envs
    for env in _envs:
        os.environ[env] = _envs[env]


def callbacks(name=None):
    """Make function as a metrics, use the same params from configuration.

    :param func: source function
    :return: wrapper
    """

    def decorator(func):
        """Provide input param to decorator.

        :param func: wrapper function
        :return: decoratpr
        """
        ClassFactory.register_cls(func, ClassType.CALLBACK_FN, alias=name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Make function as a wrapper."""
            return func(*args, **kwargs)

        return wrapper

    return decorator
