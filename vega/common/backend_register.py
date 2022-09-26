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

"""Backend Register."""

import os
import logging
import traceback

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
    devices = os.environ.get("NPU_VISIBLE_DEVICES", None) or os.environ.get("NPU-VISIBLE-DEVICES", None)
    if devices:
        os.environ['NPU_VISIBLE_DEVICES'] = devices
    # CUDA visible
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'GPU'
    elif 'NPU_VISIBLE_DEVICES' in os.environ:
        os.environ['DEVICE_CATEGORY'] = 'NPU'

    # CUDA_VISIBLE_DEVICES
    if device_category.upper() == "GPU" and "CUDA_VISIBLE_DEVICES" not in os.environ:
        if backend.lower() in ['pytorch', "p"]:
            import torch
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [str(x) for x in list(range(torch.cuda.device_count()))])
        elif backend.lower() in ['tensorflow', "t"]:
            from tensorflow.python.client import device_lib
            devices = device_lib.list_local_devices()
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [x.name.split(":")[2] for x in devices if x.device_type == "GPU"])

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

    import_extension_module()
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
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            device_id = int(os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0])
    return "{}:{}".format(device_category.lower(), device_id)


def import_extension_module():
    """Import extension module."""
    if is_npu_device():
        try:
            import ascend_automl
        except ImportError:
            logging.debug(traceback.format_exc())
