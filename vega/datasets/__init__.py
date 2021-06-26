# -*- coding:utf-8 -*-

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""Import and register datasets automatically."""

import vega
from vega.common.class_factory import ClassFactory


def Adapter(dataset):
    """Adapter of dataset."""
    if vega.is_torch_backend():
        from .pytorch.adapter import TorchAdapter as Adapter
    elif vega.is_tf_backend():
        from .tensorflow.adapter import TfAdapter as Adapter
    elif vega.is_ms_backend():
        from .mindspore.adapter import MsAdapter as Adapter
    else:
        raise ValueError
    return Adapter(dataset)


def register_datasets(backend):
    """Import and register datasets automatically."""
    if backend == "pytorch":
        from . import pytorch
    elif backend == "tensorflow":
        from . import tensorflow
    elif backend == "mindspore":
        import mindspore.dataset
        from . import mindspore
    ClassFactory.lazy_register("vega.datasets.common", {"imagenet": ["Imagenet"]})
    from . import common
    from . import transforms
