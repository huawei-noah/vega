# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Backend Register."""
from .common.class_factory import ClassFactory
from ..search_space.networks import NetworkFactory


def register_pytorch():
    """Register class factory of pytorch."""
    # register trainer
    import vega.core.trainer.pytorch
    # register evaluator
    import vega.core.evaluator.gpu_evaluator
    # register metrics
    import vega.core.metrics.pytorch
    # register algorithms
    from vega.algorithms import nas
    from vega.algorithms import hpo
    from vega.algorithms import data_augmentation
    from vega.algorithms import compression
    # reigister datasets
    import vega.datasets.pytorch
    # register networks
    import vega.search_space.networks.pytorch
    import vega.model_zoo


def register_tensorflow():
    """Register class factory of tensorflow."""
    # register trainer
    import vega.core.trainer.tensorflow
    # register datasets
    import vega.datasets.tensorflow
    # register networks
    import vega.search_space.networks.tensorflow


def set_backend(backend='pytorch'):
    """Set backend.

    :param backend: backend type, default pytorch
    :type backend: str
    """
    if backend == 'pytorch':
        register_pytorch()
    elif backend == 'tensorflow':
        register_tensorflow()
    else:
        raise Exception('backend must be pytorch or tensorflow')
