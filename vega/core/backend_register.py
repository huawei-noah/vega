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
from .common.class_factory import ClassFactory
from ..search_space.networks import NetworkFactory


def register_pytorch():
    """Register class factory of pytorch."""
    # register trainer
    import vega.core.trainer.timm_trainer_callback
    # register evaluator
    from vega.core.evaluator.evaluator import Evaluator
    from vega.core.evaluator.davinci_mobile_evaluator import DavinciMobileEvaluator
    from vega.core.evaluator.gpu_evaluator import GpuEvaluator

    # register metrics
    import vega.core.metrics.pytorch
    # reigister datasets
    import vega.datasets.pytorch
    # register networks
    import vega.search_space.networks.pytorch
    import vega.model_zoo


def register_tensorflow():
    """Register class factory of tensorflow."""
    # register metrics
    import vega.core.metrics.tensorflow
    # register datasets
    import vega.datasets.tensorflow
    # register networks
    import vega.search_space.networks.tensorflow


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
        register_pytorch()
    elif backend == 'tensorflow':
        os.environ['BACKEND_TYPE'] = 'TENSORFLOW'
        register_tensorflow()
    else:
        raise Exception('backend must be pytorch or tensorflow')
    import vega.core.trainer.trainer
    import vega.search_space.search_algs.ps_differential
    import vega.algorithms


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
