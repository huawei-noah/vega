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

"""Import and register datasets automatically."""

import vega
from vega.common.class_factory import ClassFactory


def Adapter(dataset):
    """Adapter of dataset."""
    if vega.is_torch_backend():
        from .pytorch.adapter import TorchAdapter as Adapter_backend
    elif vega.is_tf_backend():
        from .tensorflow.adapter import TfAdapter as Adapter_backend
    elif vega.is_ms_backend():
        from .mindspore.adapter import MsAdapter as Adapter_backend
    else:
        raise ValueError
    return Adapter_backend(dataset)


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
