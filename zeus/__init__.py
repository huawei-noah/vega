# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register zeus modules automatically."""

import os


def register_zeus(backend):
    """Import and register zeus modules automatically."""
    from zeus.datasets import register_datasets
    from zeus.modules import register_modules
    from zeus.networks import register_networks
    from zeus.evaluator import register_evaluator
    from zeus.trainer import register_trainer, trainer_api
    from zeus.metrics import register_metrics
    from zeus.model_zoo import register_modelzoo
    register_datasets(backend)
    register_trainer(backend)
    register_evaluator()
    register_metrics(backend)
    register_modules()
    register_networks(backend)
    register_modelzoo(backend)


def set_backend(backend='pytorch', device_category='GPU'):
    """Set backend.

    :param backend: backend type, default pytorch
    :type backend: str
    """
    if "BACKEND_TYPE" in os.environ:
        return
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
    elif 'NPU-VISIBLE-DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'NPU'
        os.environ['ORIGIN_RANK_TABLE_FILE'] = os.environ['RANK_TABLE_FILE']
        os.environ['ORIGIN_RANK_SIZE'] = os.environ['RANK_SIZE']
    if device_category is not None:
        os.environ['DEVICE_CATEGORY'] = device_category
    if backend == 'pytorch':
        os.environ['BACKEND_TYPE'] = 'PYTORCH'
    elif backend == 'tensorflow':
        os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    elif backend == 'mindspore':
        os.environ['BACKEND_TYPE'] = 'MINDSPORE'
    else:
        raise Exception('backend must be pytorch, tensorflow or mindspore')


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
