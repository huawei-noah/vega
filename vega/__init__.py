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

"""Vega's methods."""


__all__ = [
    "set_backend",
    "is_cpu_device", "is_gpu_device", "is_npu_device",
    "is_ms_backend", "is_tf_backend", "is_torch_backend",
    "get_devices",
    "ClassFactory", "ClassType",
    "FileOps",
    "run",
    "init_cluster_args",
    "module_existed",
    "TrialAgent",
    "get_network",
    "get_dataset",
    "get_trainer",
    "get_quota",
]

__version__ = "1.8.5"


import sys
if sys.version_info < (3, 6):
    sys.exit('Sorry, Python < 3.6 is not supported.')


from .common.backend_register import set_backend, is_cpu_device, is_gpu_device, is_npu_device, \
    is_ms_backend, is_tf_backend, is_torch_backend, get_devices
from .common.class_factory import ClassFactory, ClassType
from .common.file_ops import FileOps
from .core import run, init_cluster_args, module_existed
from .trainer.trial_agent import TrialAgent
from . import quota


def get_network(name, **kwargs):
    """Return network."""
    return ClassFactory.get_cls(ClassType.NETWORK, name)(**kwargs)


def get_dataset(name, **kwargs):
    """Return dataset."""
    return ClassFactory.get_cls(ClassType.DATASET, name)(**kwargs)


def get_trainer(name="Trainer", **kwargs):
    """Return trainer."""
    return ClassFactory.get_cls(ClassType.TRAINER, name)(**kwargs)


def get_quota(**kwargs):
    """Return quota."""
    return ClassFactory.get_cls(ClassType.QUOTA, "Quota")(**kwargs)
