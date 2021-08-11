# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Backend Register."""

import os
import sys

__all__ = [
    "set_backend",
    "is_cpu_device", "is_gpu_device", "is_npu_device",
    "is_ms_backend", "is_tf_backend", "is_torch_backend",
    "get_devices",
]


def set_backend(backend='pytorch', device_category='GPU'):
    """Set backend.

    :param backend: backend type, default pytorch
    :type backend: str
    """
    # if "BACKEND_TYPE" in os.environ:
    #     return
    if 'NPU_VISIBLE_DEVICES' in os.environ:
        os.environ['NPU-VISIBLE-DEVICES'] = os.environ['NPU_VISIBLE_DEVICES']
    # CUDA visible
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
    elif 'NPU-VISIBLE-DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'NPU'
        if 'RANK_TABLE_FILE' in os.environ:
            os.environ['ORIGIN_RANK_TABLE_FILE'] = os.environ['RANK_TABLE_FILE']
        os.environ['ORIGIN_RANK_SIZE'] = os.environ['RANK_SIZE']

    # device
    if device_category is not None:
        os.environ['DEVICE_CATEGORY'] = device_category.upper()
        from vega.common.general import General
        General.device_category = device_category

    # backend
    if backend.lower() in ['pytorch', "p"]:
        os.environ['BACKEND_TYPE'] = 'PYTORCH'
    elif backend.lower() in ['tensorflow', "t"]:
        os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    elif backend.lower() in ['mindspore', "m"]:
        os.environ['BACKEND_TYPE'] = 'MINDSPORE'
    else:
        raise Exception('backend must be pytorch, tensorflow or mindspore')

    # register
    from vega.datasets import register_datasets
    from vega.modules import register_modules
    from vega.networks import register_networks
    from vega.metrics import register_metrics
    from vega.model_zoo import register_modelzoo
    from vega.core import search_algs
    from vega import algorithms, evaluator
    register_datasets(backend)
    register_metrics(backend)
    register_modules()
    register_networks(backend)
    register_modelzoo(backend)

    # register ext modules
    vega_extension_path = os.environ.get("VEGA_EXTENSION_PATH")
    if vega_extension_path:
        sys.path.append(vega_extension_path)
    try:
        import vega_extension
    except ImportError:
        pass
    # backup config
    from vega.common.config_serializable import backup_configs
    backup_configs()


def is_cpu_device():
    """Return whether is cpu device or not."""
    return os.environ.get('DEVICE_CATEGORY', None) == 'CPU'


def is_gpu_device():
    """Return whether is gpu device or not."""
    return os.environ.get('DEVICE_CATEGORY', None) == 'GPU'


def is_npu_device():
    """Return whether is npu device or not."""
    return os.environ.get('DEVICE_CATEGORY', None) == 'NPU'


def is_torch_backend():
    """Return whether is pytorch backend or not."""
    return os.environ.get('BACKEND_TYPE', None) == 'PYTORCH'


def is_tf_backend():
    """Return whether is tensorflow backend or not."""
    return os.environ.get('BACKEND_TYPE', None) == 'TENSORFLOW'


def is_ms_backend():
    """Return whether is tensorflow backend or not."""
    return os.environ.get('BACKEND_TYPE', None) == 'MINDSPORE'


def get_devices():
    """Get devices."""
    device_id = os.environ.get('DEVICE_ID', 0)
    device_category = os.environ.get('DEVICE_CATEGORY', 'CPU')
    if device_category == 'GPU':
        device_category = 'cuda'
    return "{}:{}".format(device_category.lower(), device_id)
